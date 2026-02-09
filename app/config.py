from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #Document Processing Setting
    chunk_size:int=1500
    chunk_overlap:int=300

    #Collection Name
    collection_name:str="rag_documents"

    #Embedding Setting:
    embedding_model:str="text-embedding-3-small"

    #Vector Store Setting
    qdrant_api_key:str
    qdrant_url:str

    #Query and Retrieval
    openai_api_key:str
    llm_model:str = "gpt-4o-mini"
    llm_temp:float =0.0
    retieval_k:int=4


    #log setting
    log_level:str = "INFO"

    #API Setting
    api_host:str="0.0.0.0"
    api_port:int=8000


    #Ragas Evaluation
    ragas_llm_model:str|None = None
    ragas_llm_temp:float|None = None
    enble_ragas_evaluation:bool = True
    ragas_timeout_seconds:float=60.0
    ragas_log_results:bool=True
    ragas_embedding_model:str|None = None

    #Application Info
    app_name:str = "RAG Q&A System"
    app_version:str="0.1.0"

@lru_cache
def get_settings()-> Settings:
    return Settings()
    







