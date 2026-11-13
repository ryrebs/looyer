import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from json.decoder import JSONDecodeError
from pathlib import Path

import mlflow

from components import (
    build_durable_storage_context,
    build_embed_model,
    build_llm,
    build_qdrant_client,
    check_llm_server,
    irac_qa_template,
    irac_refine_template,
    rag_response_syn,
    rerank,
    rerank_score_threshold,
    retrieve,
    simpostprocessor,
)
from components.evaluator.judge import (
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    JUDGE_SYSTEM,
    judge_correctness,
    judge_faithfulness,
)
from components.llm.config import load_role
from components.synthesizer.templates import INSUFFICIENT_CONTEXT
from components.tracing.config import EXPERIMENT_EVAL, setup_tracing

_logger = logging.getLogger(__name__)

FUSED_TOP_K = 20  # candidates returned by fusion
RANK_TOP_N = 5  # of those, how many the reranker keeps (what the LLM sees)

GOLDEN_DATASET_PATH = "components/evaluator/golden_dataset.json"
RUNS_DIR = Path("components/evaluator/runs")  # one JSON file per run

# Fields that define the answer key: changing any of them changes the run's answer_key_sha256
HASHED_FIELDS = ("id", "question", "reference_answer", "source_text", "expected_metadata")
# Per-question fields the summary averages (exact names: "refused" must not match "refused_by")
METRIC_PREFIXES = ("hit_rate_", "recall_", "rr_")
METRIC_NAMES = ("refused", "faithful", "correct")
# Tracked files whose uncommitted changes can change retrieval results (git pathspecs)
CODE_PATHSPECS = (
    "components",
    "main.py",
    "Pipfile",
    ":(exclude)components/evaluator/golden_dataset.json",  # covered by answer_key_sha256
    ":(exclude)*.md",  # notes, not code
)


def source_name(source: str) -> str:
    """'assets/1987_const.html' → '1987_const.html' (the index stores the path given to run-embed)."""
    return Path(source).name


def get_gds(gds: dict):
    gds_ = {}
    for k, v in gds.items():
        verified_only = [d for d in v if d["verified"]]
        gds_[k] = verified_only
    return gds_


def model_label(name_or_path: str) -> str:
    """Readable model id for the run file.

    HF cache path '…/models--BAAI--bge-reranker-base/snapshots/2cfc18c…' → 'BAAI/bge-reranker-base@2cfc18c';
    anything else → its last path part ('bge-m3' stays 'bge-m3').
    """
    parts = Path(name_or_path).parts
    repo = next((p for p in parts if p.startswith("models--")), None)
    if repo and "snapshots" in parts:
        return repo.removeprefix("models--").replace("--", "/") + "@" + parts[-1][:7]
    return Path(name_or_path).name


def sha256_of(data) -> str:
    """SHA-256 of any JSON-serializable data, in canonical form (sorted keys), so key order doesn't matter."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_answer_key(verified: dict) -> str:
    """Fingerprint of the answer key this run is graded against.

    Only the verified entries' grading fields, so formatting or review_note edits don't change it.
    Same hash = same answer key = comparable runs.
    """
    return sha256_of(
        {group: [{f: d.get(f) for f in HASHED_FIELDS} for d in entries] for group, entries in verified.items()}
    )


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def code_version() -> dict:
    """Code the run used: the commit, plus a hash of uncommitted code changes (None if there are none).

    Same commit and same uncommitted changes = same code, even while working without committing.
    """
    uncommitted = git("diff", "HEAD", "--", *CODE_PATHSPECS)
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_sha256": sha256_of(uncommitted) if uncommitted else None,
    }


def retrieval_settings(embed_model) -> dict:
    """Everything outside the code that changes retrieval results."""
    collection = os.getenv("COLLECTION_NAME")
    return {
        "fused_top_k": FUSED_TOP_K,
        "rerank_top_n": RANK_TOP_N,
        # HyDE is on, as in main.py: its model and request settings change the dense search
        "hyde_model": os.getenv("HYDE_MODEL_NAME"),
        "hyde_settings": load_role("hyde").model_dump(),
        "embed_model": model_label(str(embed_model.model_name)),
        "reranker_model": model_label(os.getenv("RERANKED_LLM_MODEL", "")),
        "rerank_score_threshold": rerank_score_threshold(),
        # Answer step; prompts hashed so an edited template means a new run
        "answer_model": os.getenv("ANSWER_MODEL_NAME"),
        "answer_settings": load_role("answer").model_dump(),
        # Judge (a different model from the answer role)
        "judge_model": os.getenv("JUDGE_MODEL_NAME"),
        "judge_settings": load_role("judge").model_dump(),
        "judge_prompt_sha256": sha256_of([JUDGE_SYSTEM, FAITHFULNESS_PROMPT, CORRECTNESS_PROMPT]),
        "prompt_sha256": sha256_of(
            [str(irac_qa_template.message_templates), str(irac_refine_template.message_templates)]
        ),
        "collection": collection,
        # changes when documents are indexed again; a re-index that keeps the same count isn't
        # detected, so run with --force after re-indexing
        "index_points": build_qdrant_client().get_collection(collection).points_count,
    }


def create_run_fingerprint(answer_key: str, code: dict, settings: dict) -> str:
    """Same answer key + same code + same settings = same results, so no need to run again."""
    return sha256_of({"answer_key_sha256": answer_key, "code": code, "settings": settings})


def find_run(fingerprint: str) -> Path | None:
    """The newest saved run with this fingerprint, if any."""
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        with open(path, encoding="utf-8") as f:
            if json.load(f).get("fingerprint") == fingerprint:
                return path
    return None


def metric_names(rows: list) -> list[str]:
    """Names of the per-question metric fields (hit_rate_…, recall_…, rr_…).

    Every row has the same fields, so the first row is enough. No rows (a group with no
    verified questions) means no metrics.
    """
    if not rows:
        return []
    first_row = rows[0]
    return [name for name in first_row if name.startswith(METRIC_PREFIXES) or name in METRIC_NAMES]


def mean(values: list) -> float | None:
    """Average of the values, skipping None (questions with nothing to find). None if nothing is left."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def summary_name(metric: str) -> str:
    """Name of a metric's average: the average of the reciprocal rank (rr_) is the MRR (mrr_)."""
    if metric.startswith("rr_"):
        return "m" + metric
    return metric


## Targets for the averages in run["summary"] (per group and "overall"), not for single questions:
## a single question's hit rate is only 1 or 0. Rules of thumb for a legal RAG, not a published standard.
##   hit_rate_fused@20  >= 0.95  search must almost always find the provision; the reranker can't add a missing one
##   hit_rate_rerank@5  >= 0.90  (direct_lookup >= 0.95) the LLM sees only these 5: a ceiling on correct answers
##   recall_rerank@5    >= 0.90  a question that needs two provisions needs both
##   mrr_rerank@5       >= 0.80  the right provision is usually at rank 1
##   hit_rate_fused@20 - hit_rate_rerank@5 <= 0.05  a bigger gap = the reranker drops what search found
## With few questions each one moves an average a lot (20 questions: one miss = 0.05), so on a drop
## look at which questions failed. The real target: no run below the baseline run.
def summarize(rows: list) -> dict:
    """Average each metric over the questions of one group."""
    summary = {"questions": len(rows)}
    for metric in metric_names(rows):
        summary[summary_name(metric)] = mean([row[metric] for row in rows])
    # Judge replies that weren't exactly YES/NO, or failed
    summary["judge_errors"] = sum(len(row.get("judge_errors") or {}) for row in rows)
    return summary


def write_run(run: dict, started_at: datetime) -> Path:
    """Save the run as its own file, e.g. runs/2026-11-04T10-00-00Z.json."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{started_at:%Y-%m-%dT%H-%M-%SZ}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def get_metadata(nodes):
    metadata = []
    for rank, n in enumerate(nodes, 1):
        m = n.node.metadata
        metadata.append(
            {
                "rank": rank,
                "score": float(n.score) if n.score is not None else None,
                "source": source_name(m.get("source")),
                "article": m.get("article"),
                "section": m.get("section"),
            }
        )
    return metadata


def mark_matched(expected_metadata, returned_metadata):
    """Add "matched": True/False to each returned item.

    True only if the item equals one expected item on every key that expected item has
    (Constitution: source + article + section; Civil Code: source + article).
    """
    for r in returned_metadata:
        r["matched"] = any(all(r.get(k) == v for k, v in e.items()) for e in expected_metadata)
    return returned_metadata


def found(expected_metadata, returned_metadata):
    """For each expected item: is it in the returned list? (all its fields equal in one returned item)"""
    return [any(all(r.get(k) == v for k, v in e.items()) for r in returned_metadata) for e in expected_metadata]


def reciprocal_rank(returned_metadata):
    """1 / rank of the first matched item; 0.0 if nothing matched. Needs mark_matched() first."""
    first_rank = next((r["rank"] for r in returned_metadata if r["matched"]), None)
    return 1 / first_rank if first_rank else 0.0


def generate_answer(nodes_kept, question: str, answer_llm, entry_id: str) -> tuple:
    """Answer as main.py does. Returns (answer, answer_error, refused_by).

    refused_by: "gate" (no chunks, no LLM call), "model" (reply is exactly INSUFFICIENT_CONTEXT) or None.
    Errors are returned, not raised, so the run continues.
    """
    if not nodes_kept:
        return None, None, "gate"
    try:
        answer = rag_response_syn(
            nodes_kept,
            question,
            llm=answer_llm,
            text_qa_template=irac_qa_template,
            refine_template=irac_refine_template,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        answer_error = f"{type(exc).__name__}: {exc}"
        _logger.warning("%s: answer failed: %s", entry_id, answer_error)
        return None, answer_error, None
    refused_by = "model" if answer.strip() == INSUFFICIENT_CONTEXT else None
    return answer, None, refused_by


def judge_answer(judge_llm, group: str, entry: dict, nodes_kept, answer, answer_error, refused_by) -> tuple:
    """Returns (faithful, correct, judge_errors); verdicts are 1.0, 0.0 or None (not judged / error).

    faithful: judged unless refused. correct: negatives -> refused is correct (no judge call);
    answerable -> refused is wrong, otherwise judged against reference_answer.
    """
    faithful, correct, judge_errors = None, None, {}
    if answer_error is not None:
        return faithful, correct, judge_errors
    if refused_by is None:
        faithful, error = judge_faithfulness(judge_llm, nodes_kept, answer)
        if error:
            judge_errors["faithful"] = error
    if group == "negative":
        correct = 1.0 if refused_by else 0.0
    elif refused_by:
        correct = 0.0
    else:
        correct, error = judge_correctness(judge_llm, entry["question"], entry["reference_answer"], answer)
        if error:
            judge_errors["correct"] = error
    return faithful, correct, judge_errors


def evaluate(force: bool = False):
    """Run the retrieval evaluation and save it to RUNS_DIR.

    force=False: skip when a saved run has the same answer key, code and settings.
    """
    started_at = datetime.now(timezone.utc)
    embed_model = build_embed_model()

    try:
        with open(GOLDEN_DATASET_PATH, "r+", encoding="utf-8") as f:
            dst = f.read()
            golden_dataset = json.loads(dst)
    except (FileNotFoundError, ValueError, JSONDecodeError) as exc:
        _logger.warning(exc)
        golden_dataset = None

    if golden_dataset:
        datast = get_gds(golden_dataset)

        ## Skip the run when nothing that affects the results changed since a saved run
        answer_key = hash_answer_key(datast)
        code = code_version()
        settings = retrieval_settings(embed_model)
        fingerprint = create_run_fingerprint(answer_key, code, settings)
        previous = None if force else find_run(fingerprint)
        if previous:
            print(f"Nothing changed since {previous} (same answer key, code and settings): not running again.")
            print("Use --force to run anyway, e.g. after re-indexing.")
            return

        ## Same as main.py: one storage context for the whole run, built by the caller of retrieve()
        storage_context = build_durable_storage_context(os.getenv("COLLECTION_NAME"), client=build_qdrant_client())
        ## Same as main.py: HyDE on, one client for the whole run. Fail early if its server is down.
        check_llm_server("hyde")
        check_llm_server("answer")
        check_llm_server("judge")
        hyde_llm = build_llm("hyde")
        answer_llm = build_llm("answer")
        judge_llm = build_llm("judge")
        setup_tracing(EXPERIMENT_EVAL)

        results = {}
        for group, v in datast.items():
            results[group] = []
            for d in v:
                q = d["question"]
                seconds = {}  # per stage

                ## One trace per question. Spans get only plain values: an LLM client
                ## passed as a span input would be saved with its api_key.
                with mlflow.start_span(name=d["id"]) as question_span:
                    question_span.set_inputs({"question": q, "group": group})

                    ## Get metadata after node processing
                    ## to avoid overwriting scores
                    t0 = time.perf_counter()
                    nodes_fused = retrieve(
                        q, storage_context, k=FUSED_TOP_K, embed_model=embed_model, hyde_llm=hyde_llm
                    )
                    seconds["retrieve"] = round(time.perf_counter() - t0, 2)  # HyDE + dense + BM25 + fusion
                    f_metadata = mark_matched(d["expected_metadata"], get_metadata(nodes_fused))

                    t0 = time.perf_counter()
                    nodes_reranked = rerank(nodes_fused, q, top_n=RANK_TOP_N)
                    seconds["rerank"] = round(time.perf_counter() - t0, 2)
                    ## As main.py. Metrics use nodes_reranked; the LLM gets nodes_kept.
                    nodes_kept = simpostprocessor(settings["rerank_score_threshold"], nodes_reranked)

                    t0 = time.perf_counter()
                    answer, answer_error, refused_by = generate_answer(nodes_kept, q, answer_llm, d["id"])
                    seconds["answer"] = round(time.perf_counter() - t0, 2)

                    t0 = time.perf_counter()
                    with mlflow.start_span(name="judge") as span:
                        span.set_inputs({"group": group, "refused_by": refused_by})
                        faithful, correct, judge_errors = judge_answer(
                            judge_llm, group, d, nodes_kept, answer, answer_error, refused_by
                        )
                        span.set_outputs({"faithful": faithful, "correct": correct, "judge_errors": judge_errors})
                    seconds["judge"] = round(time.perf_counter() - t0, 2)

                    question_span.set_outputs({"refused_by": refused_by, "faithful": faithful, "correct": correct})

                r_metadata = mark_matched(d["expected_metadata"], get_metadata(nodes_reranked))
                found_fused = found(d["expected_metadata"], f_metadata)
                found_rerank = found(d["expected_metadata"], r_metadata)

                outmeta = {
                    "gds_id": d["id"],
                    "question": q,
                    "reference_ans": d["reference_answer"],
                    "expected_metadata": d["expected_metadata"],
                    "fused_metadata": f_metadata,
                    "rerank_metadata": r_metadata,
                    ## The reranker's best score: what RERANK_SCORE_THRESHOLD in .env is set from
                    ## (compare answerable vs negative questions)
                    "best_rerank_score": nodes_reranked[0].score if nodes_reranked else None,
                    ## Chunks left after the threshold (0 = the gate refused)
                    "kept_after_threshold": len(nodes_kept),
                    ## contains_token: token written but not alone (malformed refusal)
                    "answer": answer,
                    "answer_error": answer_error,
                    "refused_by": refused_by,
                    ## Averaged per group: negatives target 1.0, answerable target 0.0
                    "refused": None if answer_error else (1.0 if refused_by else 0.0),
                    "contains_token": bool(answer) and INSUFFICIENT_CONTEXT in answer and refused_by is None,
                    ## 1.0 YES, 0.0 NO, None not judged / judge error
                    "faithful": faithful,
                    "correct": correct,
                    "judge_errors": judge_errors,
                    "seconds": seconds,
                    ## Check if expected metadata are found
                    ## e.g. expected = [{a: 1, b: 2}, {...other}]
                    ## found = [True, False]
                    "found_fused": found_fused,
                    "found_rerank": found_rerank,
                    ## Tells if any of the expected metada were returned
                    ## None when nothing is expected (negatives), so they don't count as misses in the average
                    f"hit_rate_fused@{FUSED_TOP_K}": (1.0 if any(found_fused) else 0.0) if found_fused else None,
                    f"hit_rate_rerank@{RANK_TOP_N}": (1.0 if any(found_rerank) else 0.0) if found_rerank else None,
                    ## Tells how many were found out of the expected metadata
                    ## found = [True, False] = sum 1
                    ## len found = 2
                    ## = recall is 0.5
                    ## 50% were actually found
                    # expected provisions found ÷ expected provisions; None when nothing is expected (negatives)
                    f"recall_fused@{FUSED_TOP_K}": sum(found_fused) / len(found_fused) if found_fused else None,
                    f"recall_rerank@{RANK_TOP_N}": sum(found_rerank) / len(found_rerank) if found_rerank else None,
                    ## Reciprocal rank: 1 / rank of the first matched item (rank 2 = 0.5, none = 0.0)
                    ## Its mean over all questions is the MRR (Mean Reciprocal Rank) of the run
                    f"rr_fused@{FUSED_TOP_K}": reciprocal_rank(f_metadata) if d["expected_metadata"] else None,
                    f"rr_rerank@{RANK_TOP_N}": reciprocal_rank(r_metadata) if d["expected_metadata"] else None,
                }
                results[group].append(outmeta)

        all_rows = [row for rows in results.values() for row in rows]
        run = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "fingerprint": fingerprint,
            "answer_key_sha256": answer_key,
            "code": code,
            ## What produced these numbers: change any of these and runs aren't directly comparable
            "settings": settings,
            ## Averages per group and overall; per-question details are in "results"
            "summary": {**{group: summarize(rows) for group, rows in results.items()}, "overall": summarize(all_rows)},
            "results": results,
        }
        path = write_run(run, started_at)
        print(json.dumps(run["summary"], indent=2))
        print(f"Run saved: {path}")


if __name__ == "__main__":
    """
    We need to evaluate how correct is our RAG when we perform a query.
    For example:
        Question: "Define bill of rights?"
        Answer: "Answer here"
        Reference/Source: "Source here?
        Chunk or nodes: []

    We need to create these variables manually and
    then run our and check how close is the generated answer is
    with our manually created answer.
    """
    evaluate(force="--force" in sys.argv)
