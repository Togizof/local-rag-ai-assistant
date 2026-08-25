import sys
from src.pipeline import RAGPipeline

def main():
    """
    Simple CLI to talk with the local RAG.
    """
    print("=========================================================")
    print("           Local RAG AI Assistant - Console              ")
    print("=========================================================")
    print("Loading models, please wait...")
    
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nError: Could not start RAG Pipeline.")
        print(f"Detail: {e}")
        sys.exit(1)

    # Check if database has data
    chunk_count = pipeline.db.count_chunks()
    if chunk_count == 0:
        print("\n[WARNING] Database is empty!")
        print("Please add documents to 'data/docs/' and run 'python ingest.py'.\n")
    else:
        print(f"\nDatabase ready. Loaded {chunk_count} chunks.")

    print("\nType your question. Type 'exit' or 'quit' to close.")
    print("---------------------------------------------------------")

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "cikis", "çıkış"):
                print("Goodbye!")
                break
                
            print("Searching context and generating answer...")
            answer, sources = pipeline.query(question, top_k=3)
            
            print("\n------------------- ANSWER -------------------")
            print(answer)
            print("----------------------------------------------")
            
            if sources:
                print("\n[Sources]")
                for idx, src in enumerate(sources):
                    print(f"[{idx+1}] File: {src['document_name']} (Chunk: {src['chunk_index']})")
                    snippet = src['content'][:100].replace('\n', ' ')
                    print(f"    Text: \"{snippet}...\"")
            print("----------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nClosed.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
