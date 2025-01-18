from pydantic import BaseModel, Field
from typing import Optional, Union
from fastapi import UploadFile

class ExternalAiRequestSchema(BaseModel):
    prompt: str
    query: str
    file: Optional[UploadFile] = None  # Optional file upload

class ExternalAiResponseSchema(BaseModel):
    answer: str

class FileExtractResponse(BaseModel):
    extracted_text: str
