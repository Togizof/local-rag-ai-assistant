import sys
from src.pipeline import RAGPipeline

def main():
    """
    Interactive command line interface (CLI) to query the library assistant.
    """
    print("=========================================================")
    print("           My Library Assistant - Console UI             ")
    print("=========================================================")
    print("Loading library models, please wait...")
    
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nError: Could not start Library Pipeline.")
        print(f"Detail: {e}")
        sys.exit(1)

    # Check database chunk count before starting query loop
    chunk_count = pipeline.db.count_chunks()
    if chunk_count == 0:
        print("\n[WARNING] Library database is empty!")
        print("Please place book notes in 'data/docs/' and run 'python ingest.py'.\n")
    else:
        print(f"\nLibrary database ready. {chunk_count} book passages loaded.")

    print("\nLibrary assistant is ready! Type 'exit' or 'quit' to close.")
    print("---------------------------------------------------------")

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "cikis", "çıkış"):
                print("Exiting library assistant. Goodbye!")
                break
                
            print("Searching book notes and generating answer...")
            answer, sources = pipeline.query(question, top_k=3)
            
            print("\n------------------- ANSWER -------------------")
            print(answer)
            print("----------------------------------------------")
            
            if sources:
                print("\n[Referenced Book Chunks]")
                for idx, src in enumerate(sources):
                    print(f"[{idx+1}] Book File: {src['document_name']} (Chunk No: {src['chunk_index']})")
                    snippet = src['content'][:100].replace('\n', ' ')
                    print(f"    Passage: \"{snippet}...\"")
            print("----------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nClosed.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
