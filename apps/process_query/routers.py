from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from apps.process_query import schemas
from apps.process_query.services import ProcessQueryService

# --------- 
from typing import List
import os
import time
import re

# Define the path to your upload and static directories
UPLOAD_DIRECTORY = "uploads"

# Create directories if they don't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    
# -------- 
router = APIRouter(
    prefix="/api/v1",
    tags=["ProcessQuery"]
)

@router.post(
    "/process-query",
    status_code=status.HTTP_200_OK,
    response_model=schemas.FileUploadResponse,
    summary="Process a query with context",
    description="Process a query along with context and return a response from the selected model.",
)
async def process_query(payload: schemas.QueryPayload):
    try:
        response = await ProcessQueryService.process_query(payload)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=schemas.FileUploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    allowed_extensions = ["jpg", "jpeg", "png", "pdf", "txt"]
    max_file_size = 25 * 1024 * 1024  # 25 MB
    
    uploaded_files = []
    skipped_files = []  # List to hold skipped file paths
    
    for file in files:
        # Normalize extension to lowercase
        file_extension = file.filename.split('.')[-1].lower()

        # Check file extension
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File {file.filename} has an invalid extension.")
        
        # Check file size
        try:
            contents = await file.read()
            file_size = len(contents)
            if file_size > max_file_size:
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds the maximum size limit.")
            
            # Generate timestamp and new filename
            timestamp = int(time.time())  # Get current timestamp
            sanitized_filename = file.filename.lower()
            sanitized_filename = re.sub(r'\s+', '_', sanitized_filename)  # Replace spaces with underscores
            sanitized_filename = re.sub(r'[^a-z0-9_.-]', '', sanitized_filename)  # Remove unwanted characters
            new_filename = f"{timestamp}_{sanitized_filename}"  # Format as timestamp_filename
            file_path = os.path.join(UPLOAD_DIRECTORY, new_filename)  # New file path
            
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            file_url = f"/static/{new_filename}"  # URL accessible from the browser
            uploaded_files.append({
                "fileName": new_filename,  # Use sanitized filename
                "fileUrl": file_url, 
                "filePath": file_path
            })
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error while processing file {file.filename}: {str(e)}")
        
        finally:
            await file.close()

    return schemas.FileUploadResponse(
        status="success", 
        data={"uploadedFiles": uploaded_files, "skippedFiles": skipped_files}
    )
