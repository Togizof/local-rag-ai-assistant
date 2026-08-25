import os
import argparse
from src.pipeline import RAGPipeline

def main():
    """
    Ingest tool to read docs and save them in SQLite.
    """
    parser = argparse.ArgumentParser(description="Ingest docs into SQLite.")
    parser.add_argument("--docs_dir", type=str, default="data/docs", help="Docs folder path.")
    parser.add_argument("--clear", action="store_true", help="Clear DB before starting.")
    args = parser.parse_args()

    # Create folder if not exists
    os.makedirs(args.docs_dir, exist_ok=True)

    print("=== Local RAG Indexing ===")
    print("Loading models and starting pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nError: Could not start pipeline. Check Foundry Local status.")
        print(f"Detail: {e}")
        return

    # Clear DB if requested
    if args.clear:
        print("\nClearing old database...")
        pipeline.db.clear_database()
        print("Done.")

    # Get .txt and .md files
    supported_extensions = (".txt", ".md")
    files = [f for f in os.listdir(args.docs_dir) if f.endswith(supported_extensions)]

    if not files:
        print(f"\nNo .txt or .md files found in '{args.docs_dir}'.")
        print("Add some files and try again.")
        return

    print(f"\nIndexing {len(files)} files...")
    
    total_chunks = 0
    for file_name in files:
        file_path = os.path.join(args.docs_dir, file_name)
        chunks_added = pipeline.ingest_document(file_path)
        total_chunks += chunks_added
        print(f"-> {file_name}: saved {chunks_added} chunks.")

    print("\n=======================================")
    print("Finished indexing successfully!")
    print(f"Added Chunks: {total_chunks}")
    print(f"Total Chunks in DB: {pipeline.db.count_chunks()}")
    print("=======================================")

if __name__ == "__main__":
    main()
