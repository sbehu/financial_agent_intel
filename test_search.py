import os
import chromadb

db_path = os.path.join(os.getcwd(), "chroma_db_storage")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="amex_historical_index")

# Test a clean, distilled keyword lookup directly against the index matrix
test_query = "first line of defense erm policy"
results = collection.query(query_texts=[test_query], n_results=2)

print("\n🎯 --- RAW CHROMADB LOOKUP TEST ---")
print(f"Query: '{test_query}'")
print(f"Results Found: {len(results['documents'][0])}")
for idx, doc in enumerate(results['documents'][0]):
    print(f"\n[Result {idx+1}]\n{doc[:300]}...")