from pydantic import BaseModel
from typing import Optional, List, Dict

class ProcessQueryResponse(BaseModel):
    query: str
    retrievedContext: str
    modelResponse: str

# Pydantic models to define the structure of the response
class FileResponse(BaseModel):
    fileName: str
    fileUrl: str
    filePath: str

class FileUploadResponse(BaseModel):
    status: str
    data: dict

# Define input structure for /process-query
class QueryPayload(BaseModel):
    query: str
    context: Optional[str] = None
    files: Optional[List[dict]] = None
    model: str
    options: dict
