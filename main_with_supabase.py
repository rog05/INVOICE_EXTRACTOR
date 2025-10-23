import os
import json
import pdfplumber
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from io import BytesIO
import pytesseract
from PIL import Image
from INPRB import few_shot_single_llm_prompt
from supabase import create_client, Client
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL or SUPABASE_ANON_KEY not found in environment")

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

# Define target categories for filtering
TARGET_CATEGORIES = [
    "Core Financial Statements",
    "Source Documents", 
    "Journals & Ledgers",
    "Supporting Schedules",
    "Management Reports",
    "Statutory/Compliance"
]

# Supported file extensions
PDF_EXTENSIONS = {'.pdf'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

# -------------------------------------------------------------------------------
# Database Functions
# -------------------------------------------------------------------------------
def get_category_id(category_name: str) -> int:
    """Get category ID from category name"""
    try:
        result = supabase.table("categories").select("id").eq("name", category_name).execute()
        if result.data:
            return result.data[0]["id"]
        else:
            raise ValueError(f"Category '{category_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error getting category: {e}")

async def save_document_to_db(document_data: dict, filename: str) -> str:
    """Save document data to Supabase database"""
    try:
        # Get category ID
        category_id = get_category_id(document_data.get("category"))
        
        # Extract common fields
        document_record = {
            "category_id": category_id,
            "document_type": document_data.get("documentType"),
            "document_id": document_data.get("documentId"),
            "date": document_data.get("date"),
            "period_start": document_data.get("period", {}).get("start") if document_data.get("period") else None,
            "period_end": document_data.get("period", {}).get("end") if document_data.get("period") else None,
            "status": document_data.get("status"),
            "notes": document_data.get("notes"),
            "description": document_data.get("description"),
            "raw_json": document_data,
            "primary_currency": document_data.get("amount", {}).get("currency") if document_data.get("amount") else None,
            "total_amount": document_data.get("amount", {}).get("gross") or document_data.get("amount", {}).get("total") if document_data.get("amount") else None,
        }
        
        # Insert document record
        doc_result = supabase.table("documents").insert(document_record).execute()
        document_uuid = doc_result.data[0]["id"]
        
        # Save line items
        if document_data.get("lineItems"):
            line_items = []
            for item in document_data["lineItems"]:
                line_item = {
                    "document_id": document_uuid,
                    "line_type": "standard",
                    "description": item.get("description"),
                    "date": item.get("date"),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unitPrice"),
                    "total_amount": item.get("total"),
                    "debit_amount": item.get("debit") or item.get("cashDebit") or item.get("bankDebit"),
                    "credit_amount": item.get("credit") or item.get("cashCredit") or item.get("bankCredit"),
                    "account_name": item.get("accountDescription") or item.get("particulars"),
                    "account_number": item.get("accountNumber"),
                    "additional_data": item
                }
                # Remove None values
                line_item = {k: v for k, v in line_item.items() if v is not None}
                line_items.append(line_item)
            
            if line_items:
                supabase.table("line_items").insert(line_items).execute()
        
        # Save parties involved
        if document_data.get("partiesInvolved"):
            parties = []
            parties_data = document_data["partiesInvolved"]
            
            # Handle nested party data
            for role, party_info in parties_data.items():
                if isinstance(party_info, dict):
                    party_record = {
                        "document_id": document_uuid,
                        "role": role,
                        "name": party_info.get("name"),
                        "address": party_info.get("address"),
                        "contact_person": party_info.get("contactPerson"),
                        "phone": party_info.get("phone"),
                        "vat_number": party_info.get("vatNumber"),
                        "customer_number": party_info.get("customerNumber"),
                        "additional_info": party_info
                    }
                elif isinstance(party_info, str):
                    party_record = {
                        "document_id": document_uuid,
                        "role": role,
                        "name": party_info,
                        "additional_info": {}
                    }
                else:
                    continue
                
                # Remove None values
                party_record = {k: v for k, v in party_record.items() if v is not None}
                parties.append(party_record)
            
            if parties:
                supabase.table("parties_involved").insert(parties).execute()
        
        # Save attachments
        if document_data.get("attachments"):
            attachments = []
            for attachment in document_data["attachments"]:
                attachment_record = {
                    "document_id": document_uuid,
                    "file_name": attachment.get("fileName") or filename,
                    "file_url": attachment.get("fileUrl"),
                    "file_type": attachment.get("fileType")
                }
                # Remove None values
                attachment_record = {k: v for k, v in attachment_record.items() if v is not None}
                attachments.append(attachment_record)
            
            if attachments:
                supabase.table("attachments").insert(attachments).execute()
        else:
            # Add the processed file as an attachment
            attachment_record = {
                "document_id": document_uuid,
                "file_name": filename,
                "file_type": os.path.splitext(filename)[1].lower()
            }
            supabase.table("attachments").insert([attachment_record]).execute()
        
        return document_uuid
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error saving document: {e}")

# -------------------------------------------------------------------------------
# Text Extraction Functions
# -------------------------------------------------------------------------------
def _extract_text_from_image(file_obj, filename: str) -> str:
    """Extract text from image using Tesseract OCR"""
    try:
        # Load image
        image = Image.open(file_obj)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use Tesseract to extract text with custom config for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%&*()_+-=[]{}|;:\'\"<>?/~`^ '
        
        # Extract text
        extracted_text = pytesseract.image_to_string(image, config=custom_config)
        
        return extracted_text.strip()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR error for {filename}: {e}")
    
def _extract_text_from_pdf(file_obj, filename: str) -> str:
    try:
        with pdfplumber.open(file_obj) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extract error {filename}: {e}")

async def extract_text_from_files(
    files: List[UploadFile] = None,
    folder_path: Optional[str] = None
) -> dict:
    """Extract text from both PDF and image files"""
    extracted_texts = {}

    def _process_file(file_obj, filename: str):
        """Process a single file (PDF or image)"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in PDF_EXTENSIONS:
            extracted_text = _extract_text_from_pdf(file_obj, filename)
        elif file_ext in IMAGE_EXTENSIONS:
            extracted_text = _extract_text_from_image(file_obj, filename)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported: {PDF_EXTENSIONS.union(IMAGE_EXTENSIONS)}"
            )
        
        extracted_texts[filename] = extracted_text

    # Process uploaded files
    if files:
        for upload in files:
            # Read file content and create BytesIO object
            content = await upload.read()
            file_obj = BytesIO(content)
            _process_file(file_obj, upload.filename)

    # Process folder files
    if folder_path:
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Invalid folder path")
        
        for fname in os.listdir(folder_path):
            file_ext = os.path.splitext(fname)[1].lower()
            if file_ext in PDF_EXTENSIONS.union(IMAGE_EXTENSIONS):
                file_path = os.path.join(folder_path, fname)
                with open(file_path, "rb") as f:
                    _process_file(f, fname)

    if not extracted_texts:
        raise HTTPException(status_code=400, detail="No supported files processed")
    
    return extracted_texts

# -------------------------------------------------------------------------------
# Classification Setup
# -------------------------------------------------------------------------------
# LangChain LLM + Parser
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1,
    google_api_key=google_api_key
)

# Create output parser
parser = JsonOutputParser()

# Create the chain with format instructions
prompt_template = few_shot_single_llm_prompt.partial(
    format_instructions=parser.get_format_instructions()
)
chain = prompt_template | llm | parser

# -------------------------------------------------------------------------------
# Document Classification & Filtering  
# -------------------------------------------------------------------------------
async def classify_and_filter(texts: dict) -> List[dict]:
    async def _classify_one(filename: str, raw_text: str) -> Optional[dict]:
        try:
            result = chain.invoke({"filename": filename, "text": raw_text})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error for {filename}: {e}")

        if isinstance(result, dict) and result.get("category") in TARGET_CATEGORIES:
            return result
        return None

    docs = []
    for task_fn, task_tx in texts.items():
        doc = await _classify_one(task_fn, task_tx)
        if doc:
            docs.append(doc)

    return docs

# -------------------------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------------------------
app = FastAPI(title="Financial Document Classifier with Supabase Storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify-documents/")
async def classify_endpoint(
    files: List[UploadFile] = File(None),
    folder_path: str = None
):
    """
    Classify financial documents and save to Supabase database
    """
    if not files and not folder_path:
        raise HTTPException(status_code=400, detail="Either files or folder_path must be provided")
    
    texts = await extract_text_from_files(files, folder_path)
    classified_docs = await classify_and_filter(texts)
    
    # Save to database
    saved_documents = []
    for i, doc in enumerate(classified_docs):
        filename = list(texts.keys())[i] if i < len(texts) else f"document_{i}"
        try:
            document_uuid = await save_document_to_db(doc, filename)
            saved_documents.append({
                "document_id": document_uuid,
                "filename": filename,
                "document_type": doc.get("documentType"),
                "category": doc.get("category")
            })
        except Exception as e:
            print(f"Error saving document {filename}: {e}")
            continue
    
    return {
        "documents": classified_docs,
        "processed_files": len(texts),
        "classified_documents": len(classified_docs),
        "saved_to_database": len(saved_documents),
        "saved_documents": saved_documents
    }

@app.get("/documents/")
async def get_documents(limit: int = 10, offset: int = 0):
    """Get documents from database with pagination"""
    try:
        result = supabase.table("documents") \
            .select("*, categories(name)") \
            .range(offset, offset + limit - 1) \
            .order("created_at", desc=True) \
            .execute()
        
        return {
            "documents": result.data,
            "count": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/documents/{document_id}")
async def get_document_details(document_id: str):
    """Get full document details with related data"""
    try:
        # Get document
        doc_result = supabase.table("documents") \
            .select("*, categories(name)") \
            .eq("id", document_id) \
            .execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = doc_result.data[0]
        
        # Get line items
        line_items = supabase.table("line_items") \
            .select("*") \
            .eq("document_id", document_id) \
            .execute()
        
        # Get parties
        parties = supabase.table("parties_involved") \
            .select("*") \
            .eq("document_id", document_id) \
            .execute()
        
        # Get attachments
        attachments = supabase.table("attachments") \
            .select("*") \
            .eq("document_id", document_id) \
            .execute()
        
        return {
            "document": document,
            "line_items": line_items.data,
            "parties_involved": parties.data,
            "attachments": attachments.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/")
async def root():
    return {
        "message": "Financial Document Classifier with Supabase Storage",
        "supported_formats": {
            "pdfs": list(PDF_EXTENSIONS),
            "images": list(IMAGE_EXTENSIONS)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Tesseract
        tesseract_version = pytesseract.get_tesseract_version()
        
        # Test Supabase connection
        supabase_test = supabase.table("categories").select("count", count="exact").execute()
        
        return {
            "status": "healthy",
            "ocr_engine": "Tesseract",
            "tesseract_version": tesseract_version,
            "database": "connected",
            "categories_count": supabase_test.count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)