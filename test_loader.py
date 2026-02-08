from data.qasper_loader import QASPERLoader

loader = QASPERLoader("qasper-dev-v0.3.json")
paper_ids = loader.get_paper_ids()[:1]

docs = loader.get_documents_for_indexing(paper_ids)

print("Number of documents:", len(docs))
print("\nSample paragraph:\n")
print(docs[0].content[:500])
print("\nMetadata:\n", docs[0].meta)
   