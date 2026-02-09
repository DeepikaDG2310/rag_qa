from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger
logger=get_logger(__name__)

@lru_cache
def get_embeddings()->OpenAIEmbeddings:

    settings = get_settings()

    logger.info(f"Initializing the Embedding process using {settings.embedding_model}")


    embeddings=OpenAIEmbeddings(model=settings.embedding_model,
                                openai_api_key=settings.openai_api_key)
    
    logger.info(f"Initializing the embedding is completed")

    return embeddings


class EmbeddingSerrvice:

    def __init__(self):
        
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = self.settings.embedding_model

    def embed_query_local(self, text:str)->list[float]:

        logger.info(f"Generating the embeddings for the query {text[:50]}")

        return self.embeddings.embed_query(text) #Here embed_query() is from openai not the local function

    def embed_documents_local(self, texts:list[str])->list[list[float]]:

        logger.info(f"Generating the embeddings for the {len(texts)} document")

        return self.embeddings.embed_documents(texts) #Here embed_documents() is from openai not the local function


