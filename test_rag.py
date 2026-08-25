import sys
from unittest.mock import MagicMock

# Mock libraries if not installed
try:
    import openai
except ImportError:
    openai_mock = MagicMock()
    sys.modules['openai'] = openai_mock

try:
    import foundry_local_sdk
except ImportError:
    foundry_mock = MagicMock()
    sys.modules['foundry_local_sdk'] = foundry_mock

import unittest
import os
from src.database import DatabaseManager
from src.pipeline import RAGPipeline

class TestRAGDatabase(unittest.TestCase):
    """Test database functions."""
    
    def setUp(self):
        self.test_db_path = "data/database/test_rag.db"
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_init_db(self):
        self.assertTrue(os.path.exists(self.test_db_path))
        self.assertEqual(self.db.count_chunks(), 0)

    def test_insert_and_count(self):
        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        self.db.insert_chunk(
            document_name="test.txt",
            chunk_index=0,
            content="test content",
            embedding=mock_embedding
        )
        self.assertEqual(self.db.count_chunks(), 1)
        
        chunks = self.db.get_all_chunks()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["document_name"], "test.txt")
        self.assertEqual(chunks[0]["content"], "test content")
        self.assertEqual(chunks[0]["embedding"], mock_embedding)

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        v4 = [-1.0, 0.0, 0.0]
        
        self.assertAlmostEqual(self.db._cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(self.db._cosine_similarity(v1, v3), 0.0, places=5)
        self.assertAlmostEqual(self.db._cosine_similarity(v1, v4), -1.0, places=5)

    def test_similarity_search(self):
        self.db.insert_chunk("doc1.txt", 0, "Doc 1", [1.0, 0.0, 0.0])
        self.db.insert_chunk("doc2.txt", 0, "Doc 2", [0.0, 1.0, 0.0])
        self.db.insert_chunk("doc3.txt", 0, "Doc 3", [0.7, 0.7, 0.0])
        
        query = [0.9, 0.1, 0.0]
        results = self.db.search_similar_chunks(query, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0]["document_name"], "doc1.txt")
        self.assertEqual(results[1][0]["document_name"], "doc3.txt")


class TestRAGPipelineLogic(unittest.TestCase):
    """Test pipeline helper functions."""

    def test_chunking_sentence_aware(self):
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        pipeline_mock = RAGPipeline.__new__(RAGPipeline)
        chunks = pipeline_mock.chunk_text(text, max_chunk_chars=30, overlap_chars=10)
        
        self.assertTrue(len(chunks) > 0)
        for c in chunks:
            self.assertTrue(len(c) > 0)

if __name__ == "__main__":
    unittest.main()
