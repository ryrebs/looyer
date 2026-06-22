import os
import sys
import time
from typing import Any, Optional

from pydantic import BaseModel, Field
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.schema import TextNode
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


class ChunkProgressHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._chunk = 0
        self._t = time.time()

    def on_event_start(self, event_type, payload=None, event_id="", **kwargs):
        if event_type == CBEventType.LLM:
            self._chunk += 1
            self._t = time.time()
            print(f"[chunk {self._chunk}] generating question...", flush=True)
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type == CBEventType.LLM:
            print(f"[chunk {self._chunk}] done in {time.time() - self._t:.1f}s", flush=True)

    def start_trace(self, trace_id=None): pass
    def end_trace(self, trace_id=None, trace_map=None): pass


class ReasoningResult(BaseModel):
    answer: str = Field(description="The derived answer from the evidence.")


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "qa":

        _handler = ChunkProgressHandler()
        _cb = CallbackManager([_handler])

        _llm_qa = LlamaCPP(
            verbose=False,
            model_path=os.getenv("LLM_MODEL"),
            temperature=0.1,
            max_new_tokens=64,
            context_window=8192,
            model_kwargs={"n_threads": os.cpu_count()},
            callback_manager=_cb,
        )

        node = TextNode(
            text=(
                "Section 14. No person shall be held to answer for a criminal offense "
                "without due process of law. The accused shall enjoy the right to be "
                "heard by himself and counsel, to be informed of the nature and cause "
                "of the accusation against him, and to have a speedy, impartial, and "
                "public trial."
            )
        )

        extractor = QuestionsAnsweredExtractor(questions=1, llm=_llm_qa)

        print("\nRunning QuestionsAnsweredExtractor on 1 node...")
        t0 = time.time()
        result_qa = extractor.extract([node])
        print(f"Done in {time.time() - t0:.1f}s")
        print(f"Generated: {result_qa}")

    else:
        _llm = LlamaCPP(
            verbose=False,
            model_path=os.getenv("LLM_MODEL"),
            temperature=0.1,
            max_new_tokens=512,
            context_window=8192,
            model_kwargs={"n_threads": 5},
        )

        _program = LLMTextCompletionProgram.from_defaults(
            output_cls=ReasoningResult,
            llm=_llm,
            prompt_template_str=(
                "EVIDENCE:\n{evidence_block}\n\n"
                "QUESTION: {question}\n"
                "Answer based only on the evidence above."
            ),
            verbose=False,
        )

        result = _program(
            evidence_block="Article 1. This Act shall be known as the Civil Code of the Philippines.",
            question="What is the name of this law?",
        )
        print(result.answer)
