import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple

class DatabaseManager:
    """
    Class to manage SQLite database.
    Saves and searches text chunks and their embedding vectors.
    """
    def __init__(self, db_path: str = "data/database/rag.db"):
        self.db_path = db_path
        # Create database folder if it does not exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        # Get a new connection to SQLite
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        # Create table for chunks if it is not there
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL  -- Stored as JSON string
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def clear_database(self):
        # Delete all data in the table
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            conn.commit()
        finally:
            conn.close()

    def insert_chunk(self, document_name: str, chunk_index: int, content: str, embedding: List[float]):
        # Save a new chunk and its embedding to database
        embedding_json = json.dumps(embedding)
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (document_name, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
                (document_name, chunk_index, content, embedding_json)
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        # Get all chunks from database
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, document_name, chunk_index, content, embedding FROM chunks")
            rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunks.append({
                    "id": row["id"],
                    "document_name": row["document_name"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "embedding": json.loads(row["embedding"])
                })
            return chunks
        finally:
            conn.close()

    def count_chunks(self) -> int:
        # Count total chunks in database
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_all_document_names(self) -> List[str]:
        # Get list of unique document names in database
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT document_name FROM chunks")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        # Calculate similarity between two vectors
        try:
            import numpy as np
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            return float(dot_product / (norm_v1 * norm_v2))
        except ImportError:
            # Fallback if numpy is not installed
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_v1 = sum(a * a for a in vec1) ** 0.5
            norm_v2 = sum(b * b for b in vec2) ** 0.5
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            return dot_product / (norm_v1 * norm_v2)

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        # Find the best top_k similar chunks using brute-force search
        all_chunks = self.get_all_chunks()
        if not all_chunks:
            return []

        scored_chunks = []
        for chunk in all_chunks:
            similarity = self._cosine_similarity(query_embedding, chunk["embedding"])
            scored_chunks.append((chunk, similarity))

        # Sort by similarity score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
