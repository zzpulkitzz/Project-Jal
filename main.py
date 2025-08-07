# main.py

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager

# Import your RAG chatbot module
from rag_chatbot_module import RAGChatbotModule

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot_instance: Optional[RAGChatbotModule] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    global chatbot_instance
    try:
        logger.info("Initializing RAG Chatbot core components...")
        chatbot_instance = RAGChatbotModule(verbose=True)
        logger.info("RAG Chatbot core components initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot service...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation Chatbot Backend Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class DocumentPaths(BaseModel):
    file_paths: List[str] = Field(..., description="List of document file paths to load")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the chatbot")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None

class InitializeResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int
    ready: bool

class StatusResponse(BaseModel):
    initialized: bool
    has_documents: bool
    ready: bool
    device: str
    model_name: str
    embedding_model: str
    index_name: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ==================== Helper Functions ====================

def get_chatbot() -> RAGChatbotModule:
    """Get the global chatbot instance"""
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot service is not initialized"
        )
    return chatbot_instance

def format_sources(source_documents) -> List[Dict[str, Any]]:
    """Format source documents for frontend consumption"""
    if not source_documents:
        return []
    
    sources = []
    for i, doc in enumerate(source_documents):
        sources.append({
            "id": i + 1,
            "preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "metadata": getattr(doc, "metadata", {}),
            "relevance_score": getattr(doc, "score", None)
        })
    return sources

# ==================== API Endpoints ====================

@app.get("/", summary="Health check")
async def root():
    """Root endpoint for health checking"""
    return {
        "message": "RAG Chatbot API is running",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health", summary="Detailed health check")
async def health_check():
    """Detailed health check endpoint"""
    try:
        chatbot = get_chatbot()
        return {
            "status": "healthy",
            "ready": chatbot.is_ready(),
            "components": {
                "llm": chatbot.llm is not None,
                "embeddings": chatbot.embedding_model is not None,
                "vectorstore": chatbot.vectorstore is not None,
                "qa_chain": chatbot.qa_chain is not None
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/initialize", response_model=InitializeResponse, summary="Initialize knowledge base")
async def initialize_knowledge_base(data: DocumentPaths, background_tasks: BackgroundTasks):
    """
    Initialize the RAG chatbot with documents.
    
    This endpoint loads documents into the vector database and sets up the QA chain.
    """
    try:
        chatbot = get_chatbot()
        
        # Validate file paths exist
        missing_files = []
        for file_path in data.file_paths:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Files not found: {missing_files}"
            )
        
        # Initialize with documents
        chatbot.initialize_with_documents(data.file_paths)
        
        return InitializeResponse(
            status="success",
            message="Knowledge base initialized successfully",
            documents_loaded=len(data.file_paths),
            ready=chatbot.is_ready()
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_chatbot(request: QueryRequest):
    """
    Query the RAG chatbot with a question.
    
    Returns an answer along with source documents that were used to generate the response.
    """
    try:
        chatbot = get_chatbot()
        
        if not chatbot.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not initialized. Please call /initialize first."
            )
        
        # Query the chatbot
        response = chatbot.query(request.question)
        
        if response.get("error"):
            raise HTTPException(status_code=500, detail=response["error"])
        
        # Format response
        return QueryResponse(
            answer=response["answer"],
            sources=format_sources(response.get("source_documents", []))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/add-documents", summary="Add more documents to knowledge base")
async def add_documents(data: DocumentPaths):
    """
    Add additional documents to the existing knowledge base.
    
    This endpoint allows you to expand the knowledge base without reinitializing.
    """
    try:
        chatbot = get_chatbot()
        
        if not chatbot.has_documents:
            raise HTTPException(
                status_code=400,
                detail="Knowledge base not initialized. Use /initialize first."
            )
        
        # Validate file paths
        missing_files = []
        for file_path in data.file_paths:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Files not found: {missing_files}"
            )
        
        # Add documents
        chatbot.add_more_documents(data.file_paths)
        
        return {
            "status": "success",
            "message": f"Added {len(data.file_paths)} documents to knowledge base",
            "documents_added": len(data.file_paths)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@app.post("/upload-documents", summary="Upload and add documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents directly to the server and add them to the knowledge base.
    
    Supports .txt and .pdf files.
    """
    try:
        chatbot = get_chatbot()
        
        if not chatbot.has_documents:
            raise HTTPException(
                status_code=400,
                detail="Knowledge base not initialized. Use /initialize first or upload with /initialize-upload."
            )
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename.endswith(('.txt', '.pdf')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only .txt and .pdf files are supported."
                )
            
            file_path = upload_dir / file.filename
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_files.append(str(file_path))
        
        # Add documents to knowledge base
        chatbot.add_more_documents(saved_files)
        
        return {
            "status": "success",
            "message": f"Uploaded and added {len(files)} documents",
            "files_processed": [f.filename for f in files]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/initialize-upload", summary="Initialize with uploaded documents")
async def initialize_with_upload(files: List[UploadFile] = File(...)):
    """
    Upload documents and initialize the knowledge base in one step.
    """
    try:
        chatbot = get_chatbot()
        
        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename.endswith(('.txt', '.pdf')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            file_path = upload_dir / file.filename
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_files.append(str(file_path))
        
        # Initialize with documents
        chatbot.initialize_with_documents(saved_files)
        
        return InitializeResponse(
            status="success",
            message="Knowledge base initialized with uploaded documents",
            documents_loaded=len(files),
            ready=chatbot.is_ready()
        )
        
    except Exception as e:
        logger.error(f"Initialize with upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.get("/status", response_model=StatusResponse, summary="Get system status")
async def get_status():
    """Get the current status of the RAG chatbot system"""
    try:
        chatbot = get_chatbot()
        status_info = chatbot.get_status()
        
        return StatusResponse(**status_info)
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get status: {str(e)}")

@app.delete("/reset", summary="Reset the knowledge base")
async def reset_knowledge_base():
    """Reset the knowledge base (clear all documents)"""
    try:
        global chatbot_instance
        
        # Reinitialize the chatbot (this clears the knowledge base)
        chatbot_instance = RAGChatbotModule(verbose=True)
        
        return {
            "status": "success",
            "message": "Knowledge base has been reset"
        }
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

# ==================== Run Configuration ====================

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
