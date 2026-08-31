import logging
from foundry_local_sdk import Configuration, FoundryLocalManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manage local LLM text generation using Foundry Local SDK.
    """
    def __init__(self, model_name: str = "phi-3.5-mini"):
        self.model_name = model_name
        self.client = None
        self.model = None
        self.manager = None
        self._initialize_model()

    def _initialize_model(self):
        # Start Foundry Local Manager and load the LLM model
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
            
            # Get chat client
            logger.info("Creating chat client...")
            self.client = self.model.get_chat_client()
            logger.info("LLM manager is ready.")
        except Exception as e:
            logger.error(f"Error starting LLMManager: {e}")
            raise e

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        # Get normal response from the LLM
        try:
            self.client.settings.temperature = temperature
            self.client.settings.max_tokens = max_tokens
            
            response = self.client.complete_chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Model could not generate response. Detail: {str(e)}"
            
    def generate_response_stream(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 512):
        # Stream response from the LLM with word/phrase buffering to avoid slow char-by-char streaming on CPU
        try:
            self.client.settings.temperature = temperature
            self.client.settings.max_tokens = max_tokens
            
            response = self.client.complete_streaming_chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            buffer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content
                    # Flush buffer if it ends with punctuation, spaces, newlines, or grows too long (>=12 chars)
                    if len(buffer) >= 12 or any(c in content for c in (" ", "\n", ".", ",", "!", "?", ";", ":")):
                        yield buffer
                        buffer = ""
            
            # Flush any remaining text
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"\n[Stream Error: {str(e)}]"
