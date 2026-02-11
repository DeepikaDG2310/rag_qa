from fastapi import APIRouter, File, UploadFile, HTTPException

from app.api.schema import DocumentUploadResponse, DocumentListResponse, ErrorResponse
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService

from app.utils.logger import get_logger

logger=get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", 
             response_model=DocumentUploadResponse,
             responses={
                 400:{"model":ErrorResponse, "description":"Invalid File type"},
                 500:{"model":ErrorResponse, "description":"Processinig Error"},
             },
             summary="Upload and ignest a document",
             description="Upload a document and create chunks to store it in Vector Store",
             )

async def upload_document(file:UploadFile=File(...,description="Document to upload")):
    
    logger.info(f"Received a document to upload : {file.filename}")

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required"
        )
    
    try:

        processor = DocumentProcessor()
        chunks=processor.process_upload_file(file=file.file, filename=file.filename)

        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="No content coube be extracted from the uploaded file"
            )
        
        vector_store=VectorStoreService()
        document_ids = vector_store.add_documents(documents=chunks)

        logger.info(f" Document successfully processed"
                    f"with {len(chunks)} chunks and {len(document_ids)} documents")
        
        return DocumentUploadResponse(
            message="Document upload was proccessed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids
        )
    except ValueError as e:
        logger.warning(f"Invalid file upload : {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f" Error procesing the document {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error Processing the document {str(e)}"
        )

@router.get("/info",
            response_model=DocumentListResponse,
            summary="Get Collection Information",
            description="Get information about the collection"            
            )
async def get_collection_info()->DocumentListResponse:

    logger.info(f"Collection information is requested")

    try:
        vector_store=VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListResponse(
            collection_name=info["Collection_Name"],
            total_documents=info["points_count"],
            status=info["status"]
        )
    except Exception as e:
        logger.error(f"Error getting the collection information"
                     )

        raise HTTPException(
            status_code=500,
            detail=f"Error getting the collection information {str(e)}"
        )
    
@router.delete("/delete",
                responses={
                    200:{"description":"Collection deleted Successfully"},
                    500:{"model":ErrorResponse, "description":"Deletion Error"}
                },
                summary="Delete Entire collection",
                description="Delete all the documents in the collection. USE WITH CAUTION"
                )
async def delete_collection()->dict:
    logger.warning("Collection deletion is requested")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message":"Deletion of the collection is successfull"}
    
    except Exception as e:
        logger.error(f"Error in Deletion process")
        return ErrorResponse(
            error=500,
            detail=f"Deletion error {str(e)}"
        )
    
