from functools import lru_cache
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document




from app.config import get_settings
from app.utils.logger import get_logger
from app.core.embeddings import get_embeddings

logger = get_logger(__name__)
settings=get_settings()

EMBEDDING_DIMENSION = 1536



@lru_cache
def get_qdrant_client()-> QdrantClient:

    logger.info(f"Initiating the Qdrant client")

    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key)
    
    logger.info(f"Completed the initialization for QdrantClient")

    return client

class VectorStoreService:
    def __init__(self, collection_name:str|None=None):

        self.collection_name = collection_name or settings.collection_name
        self.embeddings = get_embeddings()
        self.client=get_qdrant_client()

        self._ensure_collection()


        """We need collection to be present before creating the vector store 
         so we included the _ensure_collection() as internal method to make sure 
         the given collection exists or create one"""

        self.vector_store = QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name=self.collection_name,

            )
        logger.info(f"Initialised the Vector Store")


    def  _ensure_collection(self):

        logger.info(f"Checing the collection {self.collection_name} is available")

        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} is available "
                        f" with {collection_info.points_count} counts")
        except UnexpectedResponse:
            logger.info(f"Creating the collection {self.collection_name}")

            self.client.create_collection(collection_name=self.collection_name,
                                          vectors_config=VectorParams(
                                              size=EMBEDDING_DIMENSION,
                                              distance=Distance.COSINE
                                          )
                                                                               
            )

            logger.info(f"Collection {self.collection_name} is created")
    
    def health_check(self)->bool:

        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check has failed with {e}")
            return False
    
    def get_collection_info(self)->dict:

        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'Collection_Name':self.collection_name,
                'points_count':collection_info.points_count,
                'Indexed Points Count':collection_info.indexed_vectors_count,
                'status':collection_info.status
            }
        except UnexpectedResponse:
            return {
                'Collection_Name':self.collection_name,
                'points_count':0,
                'Indexed Points Count':0,
                'status':'Not Found'
            }
        
    def delete_collection(self)->None:
        logger.warning(f"Deleting the collection {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"The collection {self.collection_name} is deleted")


    def add_documents(self, documents:list[Document])->list[str]:

        if not documents:
            logger.warning(f"No documents to add")
            return []
        
        logger.info(f"Adding the documents to the Vector store collection {self.collection_name}")

        ids = [str(uuid4()) for _ in documents]

        self.vector_store.add_documents(documents=documents, ids=ids)
        logger.info(f"Added the documents to the Vector store to the collection name {self.collection_name}")

        return ids
    
    def search(self, query:str, k:int|None)->list[Document]:

        k=k or settings.retieval_k

        logger.info(f"Searching for {query[:50]}...")

        retrived_docs = self.vector_store.search(query=query, k=k)

        logger.info(f"Found {len(retrived_docs)} chunks in result ")

        return retrived_docs
    
    def search_with_score(self,query:str, k:int|None)->list[tuple[Document,float]]:

        k=k or settings.retieval_k

        logger.info(f"Searching for {query[:50]}...")

        result = self.vector_store.similarity_search_with_score(query=query, k=k)

        logger.info(f"Found {len(result)} result")

        return result
    
    def get_retriever(self, k:int|None=None) -> Any:

        k=k or settings.retieval_k

        logger.info(f"Vector store as retriever..")

        return self.vector_store.as_retriever(search_type='similarity',
                                              search_kwargs={"k":k})
            