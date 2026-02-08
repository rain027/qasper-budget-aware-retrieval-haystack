import torch
from data.qasper_loader import QASPERLoader
from indexing.document_store import init_document_store
from retrieval.budget_aware_hybrid import BudgetAwareHybridRetriever
from config.defaults import *

import indexing.document_store
print("USING document_store FROM:", indexing.document_store.__file__)
print("SOURCE CODE:")
import inspect
print(inspect.getsource(indexing.document_store))


loader = QASPERLoader("qasper-dev-v0.3.json")
paper_ids = loader.get_paper_ids()[:1]
questions = loader.get_questions(paper_ids)

# 🔴 LOAD MODE
store = init_document_store(
    sql_url="sqlite:///test.db",
    faiss_index_path="faiss.index",
    create=False,
)

retriever = BudgetAwareHybridRetriever(
    document_store=store,
    embedding_model=EMBEDDING_MODEL,
    token_budget=DEFAULT_TOKEN_BUDGET,
    signal_weights=DEFAULT_SIGNAL_WEIGHTS,
    section_type_weights=SECTION_TYPE_WEIGHTS,
    use_gpu=torch.cuda.is_available(),
)

q = questions[0]
print("\nQUESTION:", q["question"])

results = retriever.retrieve(
    query=q["question"],
    filters={"paper_id": [q["paper_id"]]},
    top_k=5,
)

for i, doc in enumerate(results, 1):
    print("\n" + "=" * 60)
    print(f"Rank {i} | Score: {doc.score:.4f}")
    print("Section:", doc.meta["section_type"])
    print("Tokens:", doc.meta["word_count"])
    print(doc.content[:300])
