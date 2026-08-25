import logging
from typing import List
from foundry_local_sdk import Configuration, FoundryLocalManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Generate text embeddings using Foundry Local SDK.
    """
    def __init__(self, model_name: str = "qwen3-embedding-0.6b"):
        self.model_name = model_name
        self.client = None
        self.model = None
        self.manager = None
        self._initialize_model()

    def _initialize_model(self):
        # Start Foundry Local Manager and load the embedding model
        try:
            if FoundryLocalManager.instance is None:
                logger.info("Starting FoundryLocalManager...")
                config = Configuration(app_name="local-rag-assistant")
                FoundryLocalManager.initialize(config)
            
            self.manager = FoundryLocalManager.instance
            
            logger.info(f"Getting model: {self.model_name}")
            self.model = self.manager.catalog.get_model(self.model_name)
            if self.model is None:
                raise ValueError(f"Model not found: {self.model_name}")
            
            # Download model if not cached
            logger.info(f"Downloading model: {self.model_name}")
            self.model.download()
            
            # Load model to memory
            logger.info(f"Loading model: {self.model_name}")
            self.model.load()
            
            # Get embedding client
            logger.info("Creating embedding client...")
            self.client = self.model.get_embedding_client()
            logger.info("Embedding manager is ready.")
        except Exception as e:
            logger.error(f"Error starting EmbeddingManager: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        # Get embedding for one text
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Get embeddings for multiple texts
        if not texts:
            return []
        
        try:
            response = self.client.generate_embeddings(texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise e
