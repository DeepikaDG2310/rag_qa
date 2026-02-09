from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.vector_store import VectorStoreService
from app.api.schema import HealthResponse, ReadinessResponse
from app import __version__
from app.utils.logger import get_logger

logger=get_logger(__name__)
router=APIRouter(prefix="/health",tags=['Health'])


@router.get("", response_model= HealthResponse,
            summary= "Basic Health Check", 
            description="Returns the basic health check status of the serivce")

async def health_check()->HealthResponse:
    logger.info(f"Health check has been requested.")

    return HealthResponse(status="healthy",
                          timestamp=datetime.now(),
                          version=__version__
                          )

@router.get("/ready",response_model = ReadinessResponse,
            summary="Cloud readiness check",
            description="Check if the service is ready to handle the requests")
async def readiness_check()->ReadinessResponse:
    logger.info(f"Readiness Check has been requested")

    try:
        vector_store=VectorStoreService()
        is_ready = vector_store.health_check()

        if not is_ready:
            raise HTTPException(status_code=503,
                                detail="Vector store is not ready")
        
        collection_info=vector_store.get_collection_info()
        return ReadinessResponse(
            status="ready",
            qdrant_connected=True,
            collection_info=collection_info
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector store Readiness check failed: {e}")
        raise HTTPException(status_code=503,
                            detail=str(e)
                            )
    

