from pathlib import Path
from typing import BinaryIO
import tempfile


from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.csv import partition_csv
from unstructured.partition.text import partition_text


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from app.utils.logger import get_logger
from app.config import get_settings

#Initialize the logging
logger = get_logger(__name__)


class DocumentProcessor:
    
    SUPPORTED_EXTENTIONS = {'.pdf','.txt','.csv','.docx','.xlsx','.pptx'}

    def __init__(self, chunk_size:int|None=None, chunk_overlap:int|None=None):

        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=['\n\n','\n','.',' ',''],
            length_function = len,
        )

        logger.info(f"Document processor is initiated with chunk size {self.chunk_size}, "
                    f"chunk overlap {self.chunk_overlap}")
        
    def element_converter(self, elements :list,filename:str|Path|None=None):
            #To extract the elements text and metadata into langchain document

            doc =[ Document(page_content=e.text,
                    metadata = e.metadata.to_dict(),
                ) for e in elements if hasattr(e, "text")]
            
            if filename:
                   for d in doc:
                          d.metadata["source"]=str(filename)
            return doc


    def load_pdf(self,file_path:str | Path):

            file_path = Path(file_path)

            elements = partition_pdf(file_path,
                                     infer_table_structure=True,
                                     strategy="hi_res",
                                     extract_images_in_pdf=True,
                                     ocr_languages=['eng'],
                                     )
            
            logger.info(f" Loaded the text content from the file {file_path}")

            return elements

    def load_text(self,file_path:str|Path):

            file_path = Path(file_path)

            elements = partition_text( file_path,
                                      
                                    )
            logger.info(f" Loaded the text content from the file {file_path}")

            return elements
            
    def load_csv(self, file_path:str|Path):

            file_path = Path(file_path)

            elements = partition_csv(file_path,
                                     infer_table_structure=True,
                                     include_header=True,
                                     )
            logger.info(f" Loaded the text content from the file {file_path}")

            return elements
        
    def load_docx(self,file_path:str|Path):

            file_path = Path(file_path)

            elements=partition_docx(file_path,
                                    infer_table_structure=True,
                                    strategy="auto",
                                    
                                    )
            logger.info(f"Loaded the text content from the file {file_path}")

            return elements
        
    def load_pptx(self,file_path:str|None):

            file_path = Path(file_path)

            elements = partition_pptx(file_path,
                                      infer_table_structure=True,
                                      include_slide_notes=True,
                                      strategy="hi_res",
                                      
                                      )
            
            logger.info(f"Loaded the contents from the file {file_path}")

            return elements
    
    def load_xlsx(self,file_path:str|Path):
           
           file_path = Path(file_path)

           elements = partition_xlsx(file_path,
                                     include_header=True,
                                     infer_table_structure=True,
                                     )
           
           logger.info(f"Loaded the content from the file {file_path}")

           return elements
    
    def load_file(self,file_path:str|Path):
           
           file_path = Path(file_path)
           extension = file_path.suffix.lower()

           if extension not in self.SUPPORTED_EXTENTIONS:
                raise ValueError(
                       f"Unsupported file extenion {extension}"
                       f"Supported file extension {self.SUPPORTED_EXTENTIONS}"
                )
           
           loaders = {
                  '.pdf':self.load_pdf,
                  '.csv':self.load_csv,
                  '.txt':self.load_text,
                  '.docx':self.load_docx,
                  '.xlsx':self.load_xlsx,
                  '.pptx':self.load_pptx
           }

           return loaders[extension](file_path)
    
    def load_upload(self,file:BinaryIO, filename:str):
           
           extension=Path(filename).suffix.lower()
           if extension not in self.SUPPORTED_EXTENTIONS:
                raise ValueError(
                       f"Unsupported file extenion {extension}"
                       f"Supported file extension {self.SUPPORTED_EXTENTIONS}"
                    )
           
           with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                  tmp_file.write(file.read())
                  tmp_path=tmp_file.name

           try:
                  elements = self.load_file(tmp_path)

              #     for e in elements:
              #            e.metadata['source']=filename

                  return elements
           finally:
                  Path(tmp_path).unlink(missing_ok=True)

    def doc_splitter(self,documents:list[Document])->list[Document]:
           
           logger.info(f"Starting the chunking process with chunk size {self.chunk_size}"
                       f"chunk overlap {self.chunk_overlap}")
           
           chunks = self.text_splitter.split_documents(documents)

           logger.info(f"Created {len(chunks)} chunks ")

           return chunks
    

    def process_file(self, file_path:str|Path):
           
           elements = self.load_file(file_path)
           documents = self.element_converter(elements)
           return self.doc_splitter(documents)
    
    def process_upload_file(self,file:BinaryIO,filename:str|Path):
           
           elements = self.load_upload(file,filename)
           documents = self.element_converter(elements, filename=filename)
           return self.doc_splitter(documents)
    
    