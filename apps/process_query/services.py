import os
from fastapi import HTTPException
import logging
from apps.process_query.utils import (
    internal_model_response,
    external_model_response,
    process_pdf_files,
    embedding_model,
    retrieve_context,
    client
)
from apps.process_query.schemas import QueryPayload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessQueryService:
    
    @staticmethod
    async def process_query(payload: QueryPayload):
        try:
            # Case 1: Query with context, model, and files
            if payload.files:
                # Check if each file exists before processing
                missing_files = []
                for file in payload.files:
                    if not os.path.exists(file['filePath']):
                        missing_files.append(file['fileName'])

                if missing_files:
                    raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(missing_files)}")

                # Process PDFs and store embeddings in ChromaDB (only if files are valid)
                decoded_files = process_pdf_files(payload.files)
                logger.info(f"decoded_files: {decoded_files}")
                
                # Define collection_name based on the first file's name (like the reference code)
                collection_name = os.path.splitext(payload.files[0]['fileName'])[0]  # Using the first file's name for collection
                
                collection = client.get_collection(name=collection_name)

                # Retrieve relevant context from ChromaDB (use passed embedding model(s) if any)
                retrieved_contexts = retrieve_context(
                    payload.query, 
                    embedding_model, 
                    collection,
                    embedding_models=payload.options.get('embeddingModel', []),
                    num_results=3
                )

                # Send the summarized context along with the query to the respective model
                if payload.model in ['GEMMA']:  # Internal models
                    response = await internal_model_response(payload.query, retrieved_contexts, payload.model)
                    print('>>>> response 1 >>>>> ',response)
                else:  # External models (e.g., CHATGPT, GEMINI)
                    response = await external_model_response(payload.query, retrieved_contexts, payload.model)

                return {"status": "success", "data": {
                    "query": payload.query,
                    "retrievedContext": retrieved_contexts,
                    "modelResponse": response
                }}

            # Case 2 & Case 3 combined: Query with or without context, no files
            if payload.query:
                context = payload.context if payload.context else ""  # Use provided context or default to empty string
                if payload.model in ['GEMMA']:  # Internal models
                    response = await internal_model_response(payload.query, context, payload.model)
                    print('>>>> response 2 >>>>> ',response)
                else:  # External models (e.g., CHATGPT, GEMINI)
                    response = await external_model_response(payload.query, context, payload.model)
                return {"status": "success", "data": {"query": payload.query, "response": response}}

            # If the structure of the payload is not as expected
            raise HTTPException(status_code=400, detail="Invalid payload structure")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
