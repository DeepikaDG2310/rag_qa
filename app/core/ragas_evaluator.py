from functools import lru_cache
from typing import Any
import time
import asyncio

from datasets import Dataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGASEvaluator:
    def __init__(self):

        logger.info(f"Initializing the Evaluation")

        self.settings = get_settings()

        ragas_llm_model=self.settings.ragas_llm_model or self.settings.llm_model
        ragas_llm_temp = self.settings.ragas_llm_temp or self.settings.llm_temp
        ragas_embedding_model = (self.settings.ragas_embedding_model if self.settings.ragas_embedding_model is not None else self.settings.embedding_model)

        self.llm = ChatOpenAI(model=ragas_llm_model,
                              temperature=ragas_llm_temp,
                              api_key=self.settings.openai_api_key)
        
        self.embedding = OpenAIEmbeddings(
            model=ragas_embedding_model,
            api_key=self.settings.openai_api_key
        )

        self.metrics = [
            faithfulness,
            answer_relevancy,
        ]

        logger.info(f"Evaluation Initiazed"
                    f"Evaluation LLM: {ragas_llm_model} with {ragas_llm_temp} temperature"
                    f"Embeddings {ragas_embedding_model}"
                    f"Metircs {self.metrics}")
        
    async def aevaluate(self,
                        question:str,
                        answer:str,
                        contexts:list[str]
                        )->dict:
        logger.info(f"Starting the evaluation for the question {question[:70]}...")

        start_time=time.time()

        try:

            dataset = self._prepare_dataset(question=question,
                                            answer=answer,
                                            contexts=contexts)
            
            results = await asyncio.to_thread(
                self._evaluate_with_timeout, dataset
            )

            evaluation_time_ms = (time.time()-start_time) * 1000

            scores = {
                'faithfulness':float(results["faithfulness"]) if "faithfulness" in results else None,
                'answer_relevancy':float(results["answer_relevancy"]) if "answer_relevancy" in results else None,
                'evaluation_time_ms':round(evaluation_time_ms,2),
                'error':None
            }

            if self.settings.ragas_log_results:
                logger.info(f"Evaluation completed"
                    f"faithfulness: {scores["faithfulness"]}",
                    f"answer_relevancy:{scores["answer_relevancy"]}",
                    f"evaluation_time_ms:{scores["evaluation_time_ms"]}"
                                    )
            
            return scores
                
        except Exception as e:
            logger.error(f"Evaluation failed due to the error: {e}")
            raise

                
    def _evaluate_with_timeout(self,dataset:Dataset)->dict:

        result = evaluate(
            dataset=dataset, 
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embedding
        )

        return result.to_pandas().to_dict("records")[0]
    
    def _prepare_dataset(self,
                         question:str,
                         answer:str,
                         contexts:list[str]
                         ) -> Dataset:
        data = {
            'question':[question],
            'answer':[answer],
            'contexts':[contexts]
        }

        logger.debug(f"Prepared Dataset for evaluation for the question {question} with {len(contexts)}")

        return Dataset.from_dict(data)
    
    def _handle_evaluation_error(self, error:Exception) -> dict:

        logger.error("Evaluation can not be completed due to the error: {error}")

        return {
            'faithfulness':None,
            'answer_relevancy':None,
            'evaluation_time_ms':None,
            'error':str(error)
        }
    
    