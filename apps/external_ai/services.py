import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from io import BytesIO
from typing import Optional, Union
import openai  # or any other AI service you'd like to integrate with
from fastapi import HTTPException, UploadFile
import os

from apps.core.date_time import DateTime
from config import settings


class ExternalAiService:
    @staticmethod
    def process_file(file) -> str:
        """Process the file, extract text if image or PDF"""
        extracted_text = ""

        # Handle image (OCR)
        if file.content_type.startswith('image'):
            image = Image.open(BytesIO(file.file.read()))
            extracted_text = pytesseract.image_to_string(image)

        # Handle PDF (OCR if it contains images)
        elif file.content_type == 'application/pdf':
            reader = PdfReader(BytesIO(file.file.read()))
            for page in reader.pages:
                extracted_text += page.extract_text() or ''
                # If page has images, extract text via OCR
                if not extracted_text.strip():
                    for img in page.images:
                        image = Image.open(BytesIO(img['data']))
                        extracted_text += pytesseract.image_to_string(image)

        # Handle text files
        elif file.content_type == 'text/plain':
            extracted_text = file.file.read().decode('utf-8')

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from the file")

        return extracted_text

    @staticmethod
    def ask_ai(prompt: str, query: str, extracted_text: str) -> str:
        """Ask the AI (e.g., ChatGPT) to answer the query based on the prompt and extracted text."""
        openai.api_key = settings.OPENAI_API_KEY  # Set the OpenAI API key

        # Prepare the prompt and query
        full_prompt = f"{prompt}\n\n{extracted_text}\n\nQuery: {query}"

        try:
            response = openai.Completion.create(
                model="text-davinci-003",  # You can choose any appropriate model
                prompt=full_prompt,
                max_tokens=150
            )
            answer = response.choices[0].text.strip()
            return answer
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI Request failed: {str(e)}")


class ExternalAiServiceHelper:
    @staticmethod
    def process_request(prompt: str, query: str, file: Optional[UploadFile] = None) -> str:
        """Process the entire request including file handling and AI response."""
        if file:
            extracted_text = ExternalAiService.process_file(file)
        else:
            extracted_text = ""

        answer = ExternalAiService.ask_ai(prompt, query, extracted_text)
        return answer
