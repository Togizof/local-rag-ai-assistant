import os
import re
import logging
from typing import List, Dict, Any, Tuple, Generator
from src.database import DatabaseManager
from src.embedding import EmbeddingManager
from src.llm import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline to connect DB, Embedding and LLM.
    """
    def __init__(self, 
                 db_path: str = "data/database/rag.db",
                 embedding_model: str = "qwen3-embedding-0.6b",
                 llm_model: str = "phi-3.5-mini"):
        
        self.db = DatabaseManager(db_path=db_path)
        self.embedding_mgr = EmbeddingManager(model_name=embedding_model)
        self.llm_mgr = LLMManager(model_name=llm_model)
        
        # Simple system prompt to prevent hallucinations
        self.system_prompt = (
            "You are a helpful AI assistant. Answer the question using only the provided context.\n"
            "If you do not know the answer, say 'I don't have that information.'\n"
            "Do not make up facts.\n"
            "Include the source document name in brackets at the end of your answer if possible."
        )

    def chunk_text(self, text: str, max_chunk_chars: int = 800, overlap_chars: int = 150) -> List[str]:
        # Split text into chunks using sentence ends (.!?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If chunk is too big, save it and start a new one
            if current_length + len(sentence) > max_chunk_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Add overlap
                last_sentence = current_chunk[-1] if len(current_chunk) > 1 else ""
                if last_sentence and len(last_sentence) < overlap_chars:
                    current_chunk = [last_sentence, sentence]
                    current_length = len(last_sentence) + 1 + len(sentence)
                else:
                    current_chunk = [sentence]
                    current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def ingest_document(self, file_path: str) -> int:
        # Read a file, chunk it, get embeddings, and save to SQLite
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 0
        
        document_name = os.path.basename(file_path)
        logger.info(f"Ingesting file: {document_name}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"File is empty: {document_name}")
                return 0
                
            chunks = self.chunk_text(content)
            logger.info(f"Split {document_name} into {len(chunks)} chunks")
            
            # Get embeddings in one batch
            embeddings = self.embedding_mgr.get_embeddings(chunks)
            
            # Save chunks to database
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self.db.insert_chunk(
                    document_name=document_name,
                    chunk_index=i,
                    content=chunk,
                    embedding=emb
                )
            
            logger.info(f"Successfully indexed {document_name}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting file {document_name}: {e}")
            return 0

    def query(self, user_question: str, top_k: int = 3, temperature: float = 0.2) -> Tuple[str, List[Dict[str, Any]]]:
        # Run normal query
        # 1. Get embedding for the user question
        query_emb = self.embedding_mgr.get_embedding(user_question)
        
        # 2. Get closest chunks from database
        similar_chunks_with_scores = self.db.search_similar_chunks(query_emb, top_k=top_k)
        
        if not similar_chunks_with_scores:
            return "No documents found in the database. Please ingest some files first.", []
            
        # Format the context text
        context_parts = []
        retrieved_chunks = []
        
        for idx, (chunk, score) in enumerate(similar_chunks_with_scores):
            context_parts.append(
                f"[Source {idx+1} - File: {chunk['document_name']} (Score: {score:.4f})]\n"
                f"{chunk['content']}"
            )
            chunk_with_score = chunk.copy()
            chunk_with_score["score"] = score
            retrieved_chunks.append(chunk_with_score)
            
        context_text = "\n\n".join(context_parts)
        
        # 3. Create prompt
        user_prompt = (
            f"Use the following context to answer the question.\n\n"
            f"--- CONTEXT ---\n"
            f"{context_text}\n"
            f"---------------\n\n"
            f"Question: {user_question}\n"
            f"Answer:"
        )
        
        logger.info("Getting answer from LLM...")
        # 4. Generate response
        answer = self.llm_mgr.generate_response(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        return answer, retrieved_chunks

    def query_stream(self, user_question: str, top_k: int = 3, temperature: float = 0.2) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        # Run streaming query
        query_emb = self.embedding_mgr.get_embedding(user_question)
        similar_chunks_with_scores = self.db.search_similar_chunks(query_emb, top_k=top_k)
        
        if not similar_chunks_with_scores:
            def empty_gen():
                yield "No documents found in the database. Please ingest some files first."
            return empty_gen(), []
            
        context_parts = []
        retrieved_chunks = []
        
        for idx, (chunk, score) in enumerate(similar_chunks_with_scores):
            context_parts.append(
                f"[Source {idx+1} - File: {chunk['document_name']} (Score: {score:.4f})]\n"
                f"{chunk['content']}"
            )
            chunk_with_score = chunk.copy()
            chunk_with_score["score"] = score
            retrieved_chunks.append(chunk_with_score)
            
        context_text = "\n\n".join(context_parts)
        
        user_prompt = (
            f"Use the following context to answer the question.\n\n"
            f"--- CONTEXT ---\n"
            f"{context_text}\n"
            f"---------------\n\n"
            f"Question: {user_question}\n"
            f"Answer:"
        )
        
        stream_gen = self.llm_mgr.generate_response_stream(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        return stream_gen, retrieved_chunks
