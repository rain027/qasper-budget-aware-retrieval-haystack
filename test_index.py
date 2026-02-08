from data.qasper_loader import QASPERLoader
from indexing.document_store import init_document_store, index_documents

loader = QASPERLoader("qasper-dev-v0.3.json")
paper_ids = loader.get_paper_ids()[:1]
docs = loader.get_documents_for_indexing(paper_ids)

# 🔴 CREATE MODE
store = init_document_store(
    sql_url="sqlite:///test.db",
    create=True,
)

index_documents(store, docs,use_gpu=False,
    faiss_index_path="faiss.index",)

print("Indexed documents:", store.get_document_count())
print("FAISS index saved ✔")
