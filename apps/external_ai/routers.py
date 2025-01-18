from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from apps.external_ai.schemas import ExternalAiRequestSchema, ExternalAiResponseSchema
from apps.external_ai.services import ExternalAiServiceHelper

router = APIRouter(
    prefix="/external-ai",
    tags=["External AI"]
)

@router.post(
    "/query",
    response_model=ExternalAiResponseSchema,
    summary="Get response from AI (ChatGPT) based on prompt, query, and file",
    description="Accepts a prompt, a query, and optionally a file (image/PDF) to extract text from, then returns an AI-generated answer."
)
async def query_external_ai(request: ExternalAiRequestSchema):
    try:
        answer = ExternalAiServiceHelper.process_request(
            prompt=request.prompt,
            query=request.query,
            file=request.file
        )
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
