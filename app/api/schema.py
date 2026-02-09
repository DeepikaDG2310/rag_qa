from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

#Health Schemas

class HealthResponse(BaseModel):
    status:str=Field(...,description=" System status ")
    timestamp:datetime=Field(default_factory=datetime.now(), description="Response timestamp" )
    version:str=Field(...,description="App Version")

class ReadinessResponse(BaseModel):
    status:str=Field(...,description="Readiness Check Status")
    qdrant_connected:bool = Field(...,description="Vector store connection status")
    collection_info:dict = Field(...,description="Collection Information")

#Document Response Schemas

class DocumentUploadResponse(BaseModel):
    message:str=Field(...,description="Document upload status")
    filename:str=Field(...,description="File name for uploaded document")
    chunks_created:int=Field(...,description="Chunks created from the uploaded document")
    document_ids:list[str]=Field(...,description="Ids of the stored documents")

class DocumentInfo(BaseModel):
    source:str=Field(...,description="Source of the Document/Filename")
    metadata:dict[str, Any]=Field(default_factory=dict,description="Document Metadata")

class DocumentListResponse(BaseModel):
    collection_name:str = Field(...,description="Name of the collection")
    total_documents:int=Field(...,description="Total No. of documents in the collection")
    status:str=Field(..., description="Collection Status")
 
#Query Schemas

class QueryRequest(BaseModel):
    question:str=Field(...,
                       description="Question to search ",
                       min_length=1,
                       max_length=1000)
    
    include_source:bool=Field(default=True,
    description="Include the source documents with the answer")

    enable_evaluation:bool=Field(default=False,
    description="Enable evaluation for the answer")

    model_config = {
        'json_schema_extra':{
            'examples':[
                {
                    "question":"What is RAG",
                    "include_source":True,
                    "enable_evaluation":False
                }
            ]
        }

    }

class SourceDocument(BaseModel):

    content:str=Field(...,description="Document Content")
    metadata:dict[str,Any]=Field(...,
                                 description="Document Meatadata")
    
class EvaluationScores(BaseModel):
    faithfulness:float|None=Field(None,
                             description="Factual consistancy with the source(0-1)",
                             ge=0.0,
                             le=1.0)
    
    answer_relevancy:float|None=Field(None,
                                 description="Measure the answer is relevant to the question(0-1)",
                                 ge=0.0,
                                 le=1.0)
    
    evaluation_time_ms:float|None=Field(None,description="Time take for evaluaton process in ms")

    error:str|None=Field(None, 
    description="Error messages if any")

class QueryResponse(BaseModel):
    question:str=Field(...,
    description="Question asked")
    answer:str=Field(...,description="Answer for the question")
    sources:list[SourceDocument]|None=Field(None,description="List of source documents")
    processing_time:float=Field(...,description="Time taken by the process to answer the question")
    evaluation:EvaluationScores|None=Field(None, description="Evaluation metrics such as Faithfulness and answer_relevancy")

#Error Response Schemas

class ErrorResponse(BaseModel):
    error:str=Field(...,description="error type")
    message:str=Field(...,description="Error Message")
    detail:str=Field(..., description="Detailed error informations")

class ValidationErrorResponse(BaseModel):
    error:str=Field(default="Validation_Error", description="Error Type")
    message:str=Field(...,description="Error Message")
    errors:list[dict]=Field(...,description="Validation Errors")

    