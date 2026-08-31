import os
import argparse
from src.pipeline import RAGPipeline

def main():
    """
    Ingest tool to read book notes and save them in the library database.
    """
    parser = argparse.ArgumentParser(description="Ingest book notes into SQLite database.")
    parser.add_argument("--docs_dir", type=str, default="data/docs", help="Directory where book notes are stored.")
    parser.add_argument("--clear", action="store_true", help="Clear library database before indexing.")
    args = parser.parse_args()

    os.makedirs(args.docs_dir, exist_ok=True)

    print("=== Library Data Indexing Tool ===")
    print("Loading models and starting RAG pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nError: Could not start RAG pipeline. Check Foundry Local status.")
        print(f"Detail: {e}")
        return

    if args.clear:
        print("\nClearing library database...")
        pipeline.db.clear_database()
        print("Done.")

    supported_extensions = (".txt", ".md")
    files = [f for f in os.listdir(args.docs_dir) if f.endswith(supported_extensions)]

    if not files:
        print(f"\nNo book notes (.txt or .md) found in '{args.docs_dir}'.")
        print("Please place book summaries or notes there and try again.")
        return

    print(f"\nIndexing {len(files)} book note files...")
    
    total_chunks = 0
    for file_name in files:
        file_path = os.path.join(args.docs_dir, file_name)
        chunks_added = pipeline.ingest_document(file_path)
        total_chunks += chunks_added
        print(f"-> {file_name}: {chunks_added} passages saved.")

    print("\n=======================================")
    print("Library indexing completed successfully!")
    print(f"Total passages added: {total_chunks}")
    print(f"Total passages in library database: {pipeline.db.count_chunks()}")
    print("=======================================")

if __name__ == "__main__":
    main()
