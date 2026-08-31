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
        
        # Highly compressed system prompt to reduce prompt processing time on CPU
        self.system_prompt = (
            "You are a helpful reading assistant.\n"
            "Guidelines:\n"
            "- Answer using ONLY the provided context in the user's language (Turkish or English).\n"
            "- If not in context, say 'Bu bilgi notlarımda bulunmuyor.' or 'Not in notes'.\n"
            "- Keep answers direct, brief and well-grounded. Do not write filler text.\n"
            "- Put source filename in brackets."
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
        
        # Skip files that are already indexed in SQLite
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_name = ?", (document_name,))
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logger.info(f"File '{document_name}' is already indexed. Skipping.")
                return 0
        except Exception as e:
            logger.error(f"Error checking index status for {document_name}: {e}")
        finally:
            conn.close()

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

    def _get_library_previews(self) -> str:
        # Read the first 300 characters of each file in data/docs to build a quick summary
        docs_dir = "data/docs"
        if not os.path.exists(docs_dir):
            return "No files in library."
            
        previews = []
        files = [f for f in os.listdir(docs_dir) if f.endswith((".txt", ".md"))]
        for file_name in files[:10]:  # Limit to 10 files to avoid massive prompts on CPU
            file_path = os.path.join(docs_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    preview = f.read(200).replace("\n", " ")  # Trim preview size
                previews.append(f"Book File: {file_name}\nPreview: {preview}...")
            except Exception:
                previews.append(f"Book File: {file_name}")
        
        if len(files) > 10:
            previews.append(f"... and {len(files) - 10} more books loaded.")
            
        return "\n\n".join(previews)

    def query(self, user_question: str, top_k: int = 2, temperature: float = 0.2) -> Tuple[str, List[Dict[str, Any]]]:
        # Run normal query
        query_emb = self.embedding_mgr.get_embedding(user_question)
        similar_chunks_with_scores = self.db.search_similar_chunks(query_emb, top_k=top_k)
        
        # Filter chunks by relevance score to keep prompt context light on CPU
        relevant_chunks = [item for item in similar_chunks_with_scores if item[1] >= 0.22]
        
        # Check if similarity score is too low or DB is empty
        if not relevant_chunks:
            logger.info("Similarity score below threshold. Reading book previews for summary.")
            previews = self._get_library_previews()
            sys_prompt = (
                "You are a helpful reading assistant. The user wants to know what notes are in their library, "
                "or they asked a general question that doesn't match any specific notes. "
                "Here are the files currently in the library with their content previews:\n\n"
                f"{previews}\n\n"
                "Write a polite, friendly response in the user's language (Turkish or English).\n"
                "Briefly summarize what books/notes are in the library. Keep it short."
            )
            answer = self.llm_mgr.generate_response(
                system_prompt=sys_prompt,
                user_prompt=user_question,
                temperature=temperature
            )
            return answer, []
            
        # Format the context text for RAG
        context_parts = []
        retrieved_chunks = []
        
        for idx, (chunk, score) in enumerate(relevant_chunks):
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
        answer = self.llm_mgr.generate_response(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        return answer, retrieved_chunks

    def query_stream(self, user_question: str, top_k: int = 2, temperature: float = 0.2) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        # Run streaming query
        query_emb = self.embedding_mgr.get_embedding(user_question)
        similar_chunks_with_scores = self.db.search_similar_chunks(query_emb, top_k=top_k)
        
        # Filter chunks by relevance score to keep prompt context light on CPU
        relevant_chunks = [item for item in similar_chunks_with_scores if item[1] >= 0.22]
        
        # Check if similarity score is too low or DB is empty
        if not relevant_chunks:
            logger.info("Similarity score below threshold. Reading previews for streaming summary.")
            previews = self._get_library_previews()
            sys_prompt = (
                "You are a helpful reading assistant. The user wants to know what notes are in their library, "
                "or they asked a general question that doesn't match any specific notes. "
                "Here are the files currently in the library with their content previews:\n\n"
                f"{previews}\n\n"
                "Write a polite, friendly response in the user's language (Turkish or English).\n"
                "Briefly summarize what books/notes are in the library. Keep it short."
            )
            stream_gen = self.llm_mgr.generate_response_stream(
                system_prompt=sys_prompt,
                user_prompt=user_question,
                temperature=temperature
            )
            return stream_gen, []
            
        context_parts = []
        retrieved_chunks = []
        
        for idx, (chunk, score) in enumerate(relevant_chunks):
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
