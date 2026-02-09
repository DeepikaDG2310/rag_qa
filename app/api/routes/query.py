from datetime import datetime
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schema import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    EvaluationScores,
    ErrorResponse
)

from app.core.rag_chain import RAGChain
from app.utils.logger import get_logger

logger=get_logger(__name__)

router=APIRouter(prefix="/query", tags=["Query"])

@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400:{"model":ErrorResponse,"description":"Invalid Request"},
        500:{"model":ErrorResponse,"description":"Query Process Error"}
    },
    summary="Ask a query",
    description="Submit a question to get the AI Generated answer from the document uploaded"
)
async def query(request:QueryRequest)->QueryResponse:
    
    logger.info(f"Query Received {request.question}"
                f"Source_include: {request.include_source} and Enable Evaluation : {request.enable_evaluation}"
                )
    
    start_time = time.time()
    
    try:

        rag_chain = RAGChain()

        if request.enable_evaluation:
            
            result = await rag_chain.aquery_with_evaluator(question=request.question, include_source=request.include_source)

            sources = ([SourceDocument(content=source["content"],
                                       metadata=source["metadata"]) 
                        for source in result["sources"]]
                        if request.include_source 
                        else None)
            
            answer=result["answer"]

            evaluation=EvaluationScores(**result["evaluation"])

        elif request.include_source:

            result = await rag_chain.aquery_with_source(question=request.question)

            sources = [SourceDocument(content=source["content"],
                                      metadata=source["metadata"]) 
                                      for source in result["sources"]
                                      ]
            answer = result["answer"]
            evaluation=None
        else:
            result = await rag_chain.aquery(question=request.question)
            sources=None
            evaluation=None

        processing_time = time.time()-start_time

        logger.info(f"Query processed  in {processing_time} ms"
                    f"enable_evaliuation : {request.enable_evaluation}"
                    )
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            evaluation=evaluation
        )
    except Exception as e:
        logger.error(f"Query can not be processed")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query : {str(e)}"
        )

@router.post("/stream",
             responses={
                 400:{"model":ErrorResponse,"description":"Invalid Query Request"},
                 500:{"model":ErrorResponse,"description":"Query Processing Error"}
                 },
                 summary="Ask a question",
                 description="Submit a question for AI to generate and stream the response"
                 )
async def query_stream(request:QueryRequest)->StreamingResponse:
    
    logger.info(f"Streaming query received {request.question}")

    try:
        rag_chain = RAGChain()

        async def generate():
            try:
                for chunk in rag_chain.stream(question=request.question):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream ")
                yield f"\n\nError : str(e)"
            
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error in setting up stream")
        raise HTTPException(
            status_code=500,
            detail=f"Error in setting up stream {str(e)}"
        )
@router.post("/search",
             responses={
                 500:{"model":ErrorResponse,"description":"Search Error"}
             },
             summary="Search query",
             description="search for relevant document without getting actual answer"
             )
async def query_search(request:QueryRequest)->dict:

    logger.info(f"Query to retrieve the relevant document is requested")

    try:
        from app.core.vector_store import VectorStoreService

        vector_store = VectorStoreService()

        result=vector_store.search_with_score(query=request.queston, k=5)

        documents = [{
            "content": doc.page_content,
            "metadata":doc.metadata,
            "relevance_score":round(score,4)
        } for doc,score in result]
        
        return {
            "query":request.question,
            "relevant_document":documents,
            "count":len(documents)
            }
    
    except Exception as e:
        logger.error(f"Error in search")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents {str(e)}"
        )
    
