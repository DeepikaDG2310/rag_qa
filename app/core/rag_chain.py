

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import get_settings
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


RAG_PROMPT_TEMPLATE = """You are an assistant. Answer the question based on the provided context.
If you don't find the answer from the provided context, say "I don't have enough information to answer the question"
Do not make up any information. Only use the given context to answer the question

context:{context}

Question:{question}

Answer:
"""

def format_documents(docs:list[Document])->str:

    return "\n\n---\n\n".join(doc.page_content for doc in docs)

class RAGChain:

    def __init__(self, vector_store:VectorStoreService|None=None):

        """context|prmpt|llm|str"""

        self.vector_store = vector_store or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        self.prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temp,
            api_key=settings.openai_api_key
        )

        self.chain = (
            {"context":self.retriever|format_documents,
             "question":RunnablePassthrough(),}
             |self.prompt
             |self.llm
             |StrOutputParser()
             
             )
        
        #To Initialize the evaluator
        self._evaluator=None

        logger.info(f"RAG chain initialized with LLM Model {settings.llm_model} "
                    f"with the top {settings.retieval_k}")
        
    
    @property
    def evaluator(self):

        if self._evaluator is None:
            from app.core.ragas_evaluator import RAGASEvaluator

            self._evaluator=RAGASEvaluator()

        return self._evaluator
    
    def query(self,question:str)->str:

        logger.info(f"Processing the question {question[:70]}...")

        try:
            answer = self.chain.invoke(question)
            logger.info(f"Query is processed")
            return answer
        except Exception as e:
            logger.error("Can not process the query due to {e}")
            raise

    def query_with_source(self,question:str)->dict:

        logger.info(f"Processing the query {question[:70]}...")

        try:
            answer = self.chain.invoke(question)

            source_docs = self.retriever.invoke(question)

            sources= [
                {
                    'content':doc.page_content,
                    'metadata':doc.metadata
                }
                for doc in source_docs
            ]
            logger.info(f"Processed the query with {len(sources)} sources")

            return {
                'answer':answer,
                'sources':sources
            }
        except Exception as e:
            logger.error(f"Can not process the query due to {e}")
            raise

    async def aquery(self, question:str)->str:
        logger.info(f"Processing the query {question[:70]}...")

        try:
            answer = await self.chain.ainvoke(question)

            logger.info(f"Processed is completed")

            return answer
        
        except Exception as e:
            logger.error(f"Can not process the query due to str{e}")
            raise

    async def aquery_with_source(self, question:str)->dict:

        logger.info(f"Processing the query {question[:70]}...")

        try:
            answer = await self.chain.ainvoke(question)

            source_docs = self.retriever.invoke(question)

            sources = [ 
                {
                    'content':doc.page_content,
                    'metadata':doc.metadata
                } 
                for doc in source_docs
            ]

            logger.info(f"Process is completed with {len(sources)} context sources")

            return {
                'answer':answer,
                'sources':sources
            }


        except Exception as e:
            logger.error(f"Can not process the query due to {e}")
            raise
    
    async def aquery_with_evaluator(self, question:str, include_source:bool=True)->dict:

        logger.info(f"Processing the query {question[:70]}...")

        try:
            result = await self.aquery_with_source(question)

            answer = result['answer']
            sources = result['sources']

            contexts = [source['content'] for source in sources]

            try:
                evaluation = await self.evaluator.aevaluate(question=question, answer=answer, contexts=contexts)

                logger.info(f"Evaluation complete"
                            f"Faithfulness {evaluation.get('faithfulness', 'N/A')}"
                            f"Answer Relavancy {evaluation.get('answer_relavancy','N/A')}"
                            )
            except Exception as e:
                logger.warning(f"Evaluation Failed due to {e}")
                evaluation={
                    "faithfulness":None,
                    "answer_relavancy":None,
                    "evaluation_time_ms":None,
                    "error":str(e)
                }
            return {'answer':answer, 
                'sources':sources,
                'evaluation':evaluation
                }
        except Exception as e:
            logger.error(f"Can not process the query {question[:70]}...")
            raise

    def stream(self,question:str):

        logger.info(f"Processing the query {question[:70]}...")

        try:
            for chunks in self.chain.stream(question):
                yield chunks
        except Exception as e:
            logger.error(f"Can not process the query due to {e}")
            raise
        