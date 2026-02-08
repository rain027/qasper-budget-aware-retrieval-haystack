# main.py

from data.qasper_loader import QASPERLoader
from indexing.document_store import init_document_store, index_documents
from retrieval.budget_aware_hybrid import BudgetAwareHybridRetriever
from experiments.runner import ExperimentRunner
from config.defaults import *
import torch


def main():
    loader = QASPERLoader("qasper-dev-v0.3.json")
    paper_ids = loader.get_paper_ids()[:10]

    docs = loader.get_documents_for_indexing(paper_ids)
    questions = loader.get_questions(paper_ids)

    store = init_document_store()
    index_documents(store, docs, use_gpu=torch.cuda.is_available())

    retriever = BudgetAwareHybridRetriever(
        document_store=store,
        embedding_model=EMBEDDING_MODEL,
        token_budget=DEFAULT_TOKEN_BUDGET,
        signal_weights=DEFAULT_SIGNAL_WEIGHTS,
        section_type_weights=SECTION_TYPE_WEIGHTS,
        use_gpu=torch.cuda.is_available(),
    )

    runner = ExperimentRunner(questions)
    print(runner.run(retriever))


if __name__ == "__main__":
    main()
