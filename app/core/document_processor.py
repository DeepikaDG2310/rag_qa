import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:

    SUPPORTED_EXTENSIONS= {'.pdf','.txt','.csv'}

    def __init__(self, chunk_size:int|None=None,
                 chunk_overlap:int|None=None):
        
        settings= get_settings()

        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',' ',''],
                                                            chunk_size= self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap,
                                                            length_function=len,)
        logger.info(f"Document Processor is initiated with chunk size = {self.chunk_size}"
                    f"chunk overlap {self.chunk_overlap}")
        
    def load_pdf(self, file_path:str|Path)->list[Document]:

        file_path = Path(file_path)
        logger.info(f"Loading the pdf {file_path}")

        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()

        logger.info(f"The file {file_path} is processed with {len(documents)}")

        return documents
    
    def load_text(self,file_path:str|Path)->list[Document]:

        file_path = Path(file_path)

        logger.info(f"Loading the file {file_path}")

        loader = TextLoader(file_path=file_path, encoding='utf-8')

        documents = loader.load()

        logger.info(f"The file {file_path} is loaded with {len(documents)} documents")

        return documents
    
    def load_csv(self,file_path:str|Path)->list[Document]:

        file_path=Path(file_path)

        logger.info(f"Loading the file {file_path}")

        loader=CSVLoader(file_path=file_path)

        documents = loader.load()

        logger.info(f"The file {file_path} is processed with {len(documents)} documents")

        return documents
    
    def load_file(self,file_path:str|Path)->list[Document]:

        file_path=Path(file_path)

        logger.info(f"Loading the file {file_path}")

        file_extension = file_path.suffix.lower()

        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extensions {file_extension}"
                f"Supported file extensions are {self.SUPPORTED_EXTENSIONS}"
            )
        
        loader = {
            '.pdf':self.load_pdf,
            '.txt':self.load_text,
            '.csv':self.load_csv
        }

        return loader[file_extension](file_path=file_path)
    
    def load_upload(self,
                    file:BinaryIO,
                    filename:str|Path)->list[Document]:
        
        file_extension = Path(filename).suffix.lower()

        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"unsupported file extension {file_extension}"
                f"Supported extensions are {self.SUPPORTED_EXTENSIONS}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=file_extension) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
            tmp_file.close()

            try:
                documents = self.load_file(file_path=tmp_path)

                for doc in documents:
                    doc.metadata['source'] = filename

                return documents
            
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def split_documents(self,documents:list[Document])->list[Document]:
        logger.info(f"Starting the document split with chunk size{self.chunk_size}"
                    f"chunking overlap {self.chunk_overlap}")
        
        chunks=self.text_splitter.split_documents(documents=documents)

        logger.info(f"chunking is completed with {len(chunks)}")

        return chunks

    def process_file(self,file_path:str|Path)->list[Document]:

        documents = self.load_file(file_path=file_path)
        return self.split_documents(documents=documents)
    
    def procee_upload_file(self,file:BinaryIO,filename:str|Path)->list[Document]:

        documents = self.load_upload(file=file,filename=filename)
        return self.split_documents(documents=documents)
    
    




    

