from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager 

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.api.routes import health, query, documents
from app.utils.logger import get_logger, set_logger

settings = get_settings()

@asynccontextmanager
async def lifespan(app:FastAPI):

    set_logger(log_level=settings.log_level)
    logger=get_logger(__name__)

    logger.info(f"Starting the application {settings.app_name} v{__version__}"
                f"Log Level : {settings.log_level}")
    
    yield

    logger.info(f"Shutting down the application")

app=FastAPI(title=settings.app_name,
            description="""
            RAG Q&A System API

A Retrieval-Augmented Generation (RAG) powered question-answering platform designed for accurate, transparent, and scalable knowledge retrieval.

Built using FastAPI, LangChain, Qdrant Cloud, and OpenAI, this system enables users to query uploaded documents and receive reliable, context-aware answers grounded in source data.

Key Capabilities

Upload and index PDF, TXT, and CSV documents

Ask natural-language questions over your content

Retrieve source-backed answers to reduce hallucinations

Stream responses in real time for a better user experience

Production-ready API with health and readiness checks 
"""
,
version=__version__,
doc_url="/docs",
redoc_url="/redoc",
openapi_url="/openapi.json",
lifespan=lifespan,
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.mount("/static", StaticFiles(directory="static"),name="static")

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    with open("static/index.html","r") as f:
        return f.read()
    
@app.exception_handler(Exception)
async def global_exception_handler(request:Request, exc=Exception):

    logger=get_logger(__name__)

    logger.error(f"Unhandled Errors: {exc}", exc_info=True)

    return JSONResponse(
            status_code=500,
            content={
                "error":"Internal Server Error",
                "message": str(exc)
                }
    )

if __name__=="__main__":
    import uvicorn

   
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
