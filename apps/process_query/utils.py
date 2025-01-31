import os
import re
import fitz  # PyMuPDF
import httpx
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from apps.core.date_time import DateTime
import nltk
import numpy as np
from fastapi import HTTPException
import chromadb
from chromadb.errors import NotFoundError  # Correctly handle the error in ChromaDB
import requests
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to your upload and static directories
UPLOAD_DIRECTORY = "uploads"

# Create directories if they don't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    
# ChromaDB client setup
client = chromadb.PersistentClient(path='./chroma_db')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean up text and remove unnecessary spaces
def clean_text(text):
    # Remove extra spaces, newlines, and ensure words are properly spaced
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    return cleaned_text

# Function to handle PDF files and return context embeddings
def process_pdf_files(files):
    decoded_files = []
    for file in files:
        try:
            document_id = file['fileName']  # Using file name as the unique ID
            
            # Dynamically create a collection based on the file name (without extension)
            collection_name = os.path.splitext(file['fileName'])[0]  # Remove file extension to use the base name as collection name
            logger.info(f"Processing file {file['fileName']} for collection: {collection_name}")

            # Check if the collection exists, and if not, create it
            logger.info(f"Checking if collection {collection_name} exists.")
            collections = client.list_collections()  # This lists all collections in the database
            if collection_name not in collections:
                logger.info(f"Collection {collection_name} not found. Creating a new collection.")
                file_collection = client.create_collection(name=collection_name)
            else:
                logger.info(f"Collection {collection_name} found.")
                file_collection = client.get_collection(name=collection_name)
            
            # Normalize file path using os.path.join to avoid platform-specific issues
            file_path = os.path.join(UPLOAD_DIRECTORY, file['fileName'])  # Use os.path.join for cross-platform compatibility
            logger.info(f"Reading file from path: {file_path}")

            # Check if the file exists before processing
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File {file['fileName']} not found at {file_path}")

            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Extract text from PDF
            extracted_text = extract_text_from_pdf(file_content)
            
            # Split text into sentences using nltk
            sentences = nltk.sent_tokenize(extracted_text)  # This will split the text into sentences

            # Clean each sentence before adding it to ChromaDB
            cleaned_sentences = [clean_text(sentence) for sentence in sentences]
            
            # Generate embeddings for each sentence
            embeddings = embedding_model.encode(cleaned_sentences)
            flattened_embeddings = [embedding.tolist() for embedding in embeddings]  # Convert ndarray to list of floats

            # Add documents to ChromaDB collection
            logger.info(f"Adding {len(sentences)} sentences to collection {collection_name}")
            file_collection.add(
                documents=cleaned_sentences,
                metadatas=[{"file_name": file['fileName']}] * len(cleaned_sentences),  # Use the same file name metadata for all sentences
                embeddings=flattened_embeddings,
                ids=[f"{document_id}_{i}" for i in range(len(cleaned_sentences))]  # Unique ID for each sentence
            )

            # Store the decoded text for response
            decoded_files.append(extracted_text)
        
        except Exception as e:
            logger.error(f"Error processing file {file['fileName']}: {e}")
            raise HTTPException(status_code=422, detail=f"Error processing file {file['fileName']}: {e}")
    
    return decoded_files


# Function to extract text from PDF using PyMuPDF and clean the result
def extract_text_from_pdf(file_content):
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract text as plain text
        # Clean the extracted text to remove unwanted spaces or characters
        cleaned_text = clean_text(text)
        return cleaned_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=422, detail=f"Error extracting text from PDF: {e}")


# Function to retrieve relevant context from ChromaDB
def retrieve_context(query, embedding_model, collection, embedding_models=None, num_results=3):
    try:
        # Encode the query into embeddings
        query_embedding = embedding_model.encode([query])
        
        # Query ChromaDB for the most relevant documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=num_results
        )
        
        # Log the results to understand the structure of the response
        logger.info(f"ChromaDB query results: {results}")

        # Check if 'documents' key exists
        if 'documents' in results:
            # Flatten the nested list structure (extract the first element of each inner list)
            # contexts = [item[0] for item in results['documents']]  # Flatten the list
            contexts = [sentence for sublist in results['documents'] for sentence in sublist]
        else:
            logger.error("ChromaDB response does not contain 'documents' key.")
            raise HTTPException(status_code=500, detail="Unexpected response structure from ChromaDB.")
        
        # If embedding models are provided, we can summarize the context from multiple sources
        if embedding_models and len(embedding_models) > 1:
            summarized_context = summarize_context(contexts)
            return summarized_context
        
        # Return the most relevant context, or an empty string if none is found
        return contexts[0] if contexts else ""
    
    except Exception as e:
        logger.error(f"Error retrieving context from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context from ChromaDB: {e}")


# Function to clean up context and remove extra spaces
def clean_context(context):
    # Remove unwanted characters (if any) like multiple spaces between words, or between characters
    cleaned = re.sub(r'\s+', ' ', context)  # Replace multiple spaces with a single space
    cleaned = cleaned.strip()  # Remove leading and trailing spaces
    return cleaned

# Function to summarize context (combine multiple pieces of context)
def summarize_context(contexts):
    # Combine the contexts and clean up extra spaces
    combined_context = " ".join(contexts)
    return clean_context(combined_context)

async def call_gemini_api(query: str, context: str) -> str:
    try:
        GEMINI_API_URL = os.getenv("GEMINI_API_URL")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        # Prepare the request payload with context and query
        payload = {
            "contents": [{"parts": [{"text": f"Context: {context}\n\nQuery: {query}"}]}]
        }

        # Set headers
        headers = {"Content-Type": "application/json"}

        # Prepare the URL with API key
        params = {"key": GEMINI_API_KEY}

        async with httpx.AsyncClient() as client:
            # Make the request to Gemini API
            response = await client.post(GEMINI_API_URL, json=payload, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response JSON data
            data = response.json()

            # Extract the generated content from the response
            generated_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", None)
            
            if generated_text:
                return generated_text
            else:
                raise HTTPException(status_code=400, detail="Gemini couldn't generate a response for your query.")
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def call_openai_api(query: str, context: str) -> str:
    try:
        OPENAI_API_URL = os.getenv("OPENAI_API_URL")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Prepare the request payload with context and query
        payload = {
            "model": "gpt-4",  # Or another model like gpt-3.5-turbo
            "messages": [
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": query}
            ]
        }

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        async with httpx.AsyncClient() as client:
            # Make the request to OpenAI API
            response = await client.post(OPENAI_API_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response JSON data
            data = response.json()

            # Extract the generated content from the response
            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", None)

            if generated_text:
                return generated_text
            else:
                raise HTTPException(status_code=400, detail="OpenAI couldn't generate a response for your query.")
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def call_gemma_api(query: str, context: str):
    try:
        model_path = os.path.join(os.getcwd(), "ai_models", "gemma-2-9b-it-bnb-4bit-si_split", "unsloth.Q4_K_M.gguf")
        print("model_path >>> ", model_path)
        
        # Load the local Llama model (adjust the model path to your actual model file)
        llm = Llama(model_path)

        # Refined system prompt
        user_prompt = f"""
        You are a helpful assistant. Given the context, answer the following question clearly and concisely.
        
        Context: {context}
        Question: {query}
        
        Answer:
        """

        # Send the prompt to the model and get the response
        response = llm.create_completion(prompt=user_prompt)
        generated_text = response["choices"][0]["text"]

        print("generated_text >>> ", generated_text)

        if not generated_text:
            logging.warning("No meaningful text found in response!")
            raise HTTPException(status_code=400, detail="No valid response generated.")

        # Return the cleaned response
        return {"generated_text": generated_text.strip()}

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")
     
# INTERNAL function: Local LLM models (e.g., GEMINI)
async def internal_model_response(query, context, model_name):
    try:
        if model_name == 'GEMMA':
            response = await call_gemma_api(query, context)
            return response
        else:
            return f"Unknown internal model: {model_name}"
    except Exception as e:
        logger.error(f"Error generating internal model response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating internal model response: {e}")

# EXTERNAL function: External AI models (e.g., CHATGPT, GEMINI)
async def external_model_response(query, context, model_name):
    try:
        if model_name == 'CHATGPT':
            response = await call_openai_api(query, context)
            return response
        elif model_name == 'GEMINI':
            response = await call_gemini_api(query, context)
            return response
        else:
            return f"Unknown external model: {model_name}"
    except Exception as e:
        logger.error(f"Error generating external model response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating external model response: {e}")


def retrieve_context_from_chroma(query, collection, embedding_models=None, num_results=3):
    try:
        query_embedding = embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding, n_results=num_results)
        if "documents" in results:
            contexts = [sentence for sublist in results["documents"] for sentence in sublist]
            return contexts[0] if contexts else ""
        raise HTTPException(status_code=500, detail="ChromaDB response does not contain 'documents'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context from ChromaDB: {str(e)}")

