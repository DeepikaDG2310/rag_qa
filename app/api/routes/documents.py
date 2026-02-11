from fastapi import APIRouter, File, UploadFile, HTTPException

from app.api.schema import DocumentUploadResponse, DocumentListResponse, ErrorResponse

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService

from app.utils.logger import get_logger
logger = get_logger(__name__)
router=APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload",
        response_model=DocumentUploadResponse,
        responses={
            400:{"model":ErrorResponse,"description":"Invalid file type"},
            500:{"model":ErrorResponse,"description":"Processin Error"}
        },
        summary="Upload and digest a document",
        description="Upload a document  to be processed and added to the vector store",
        )
async def upload_document(file:UploadFile=File(...,description="File to be processed"))->DocumentUploadResponse:

    logger.info(f"Received a file to be processed {file.filename}")

    if not file.filename:
        logger.error(f"File name is missing which is required")
        raise HTTPException(
            status_code=400,
            detail="File name is required"
        )
    
    try:
        document_processor = DocumentProcessor()
        chunks = document_processor.procee_upload_file(file=file.file, filename=file.filename)

        if not chunks:
            logger.error(f"Error: No chunks can be exracted ")
            raise HTTPException(
                status_code=400,
                detail="No chunks could be extracted from the file"
            )
        
        vector_store = VectorStoreService()
        document_ids = vector_store.add_documents(chunks)

        logger.info(f"Successfully processed the file {file.filename}"
                    f"{len(chunks)} were extracted from the file")
        
        return DocumentUploadResponse(
            message="Document upload is processed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids
        )
    except ValueError as e:
        logger.error(f"Invalid file upload")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing the file {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in processing the file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in processing the file {str(e)}"
        )

@router.get("/info",
            response_model=DocumentListResponse,
            summary="Get Collection Information",
            description="Get the Information about the collection")
async def get_collection_info()->DocumentListResponse:
    logger.info(f"Collection info is requested")

    try:
        vector_store=VectorStoreService()
        info=vector_store.get_collection_info()

        return DocumentListResponse(collection_name=info["Collection_Name"],
                                    total_documents=info["points_count"],
                                    status = info["status"]
                                    )
    except Exception as e:
        logger.error(f"Error in getting the collection information {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Getting error collection info {str(e)}"
        )
    
@router.delete("/collection",
               responses={
                   200:{"description":"Collection deletion successfull"},
                   500:{"model":ErrorResponse,"decription":"Deletion Error"}
               }
               )
async def delete_collection()->dict:

    logger.warning(f"Collection deletion is requsted")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message":"The collection is deleted successfully"}
    except Exception as e:
        logger.error(f"The deletion of the collection is not successfull: {str(e)}")
        raise HTTPException(status_code=500,
        detail=f"Error deleting the collection {str(e)}")
