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

    ## Improve or replace
    GOLDEN_DATASET_EXAMPLE = [
    {
        "question": "What rights does an accused person have during custodial investigation?",
        "reference_answer": (
            "The accused has the right to remain silent, the right to have "
            "competent and independent counsel preferably of his own choice, "
            "and the right to be informed of these rights. "
            "(Article III, Section 12, 1987 Constitution)"
        ),
        "source": "const_1987.html",
        "section": "Article III, Section 12",
        "expected_node_ids": [],
    },
]