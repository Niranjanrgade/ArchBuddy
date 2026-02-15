from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
import sys

def verify():
    print("Verifying Azure Docs Embeddings...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma(
            collection_name="AzureDocs",
            persist_directory="./chroma_db_AzureDocs",
            embedding_function=embeddings
        )
        
        count = db._collection.count()
        print(f"Total documents in DB: {count}")
        
        if count == 0:
            print("Database is empty.")
            return

        print("\nTesting similarity search for 'App Service scaling':")
        results = db.similarity_search("How to scale Azure App Service?", k=3)
        for i, res in enumerate(results, 1):
            meta = res.metadata
            print(f"\n[Result {i}]")
            print(f"Source: {meta.get('source', 'unknown')}")
            print(f"Service: {meta.get('service', 'unknown')}")
            print(f"Content snippet: {res.page_content[:150]}...")
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    verify()
