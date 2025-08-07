import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Updated imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import platform
import logging

# Pinecone import
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbotModule:
    """
    RAG Chatbot Module that can be imported and used externally.
    
    Usage:
        from rag_chatbot import RAGChatbotModule
        
        # Initialize
        chatbot = RAGChatbotModule()
        
        # Load documents
        chatbot.initialize_with_documents(["document1.txt", "document2.pdf"])
        
        # Query
        response = chatbot.query("What is the main topic?")
    """
    
    def __init__(self, 
                 index_name: str = "rag-chatbot",
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 retrieval_k: int = 3,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 verbose: bool = True):
        """
        Initialize RAG Chatbot Module
        
        Args:
            index_name: Name of the Pinecone index
            model_name: HuggingFace model name for LLM
            embedding_model: HuggingFace embedding model name
            chunk_size: Text chunk size for document splitting
            chunk_overlap: Overlap between text chunks
            retrieval_k: Number of documents to retrieve
            max_tokens: Maximum tokens for LLM generation
            temperature: Temperature for LLM generation
            verbose: Whether to print setup messages
        """
        # Configuration
        self.index_name = index_name
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Components
        self.embedding_model = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.pc = None
        self.device = self.get_device()
        
        # Initialization state
        self.is_initialized = False
        self.has_documents = False
        
        # Setup core components
        self._setup_core_components()
    
    def get_device(self):
        """Determine the best device to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            if platform.system() == "Darwin":
                version = platform.mac_ver()[0]
                major_version = int(version.split('.')[0])
                if major_version >= 14:
                    if self.verbose:
                        logger.info("Using MPS device (macOS 14.0+)")
                    return "mps"
                else:
                    if self.verbose:
                        logger.info("MPS available but using CPU due to compatibility issues")
                    return "cpu"
            else:
                return "mps"
        else:
            return "cpu"
    
    def _setup_core_components(self):
        """Initialize core components (Pinecone, embeddings, LLM)"""
        try:
            if self.verbose:
                logger.info(f"Setting up RAG Chatbot Module using device: {self.device}...")
            
            # Set MPS fallback
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            self._setup_pinecone()
            self._setup_embeddings()
            self._setup_llm()
            
            self.is_initialized = True
            if self.verbose:
                logger.info("RAG Chatbot Module core components initialized successfully!")
                
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise
    
    def _setup_pinecone(self):
        """Initialize Pinecone vector database"""
        self.pc = PineconeClient(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        if self.verbose:
            logger.info(f"Pinecone index '{self.index_name}' is ready")
    
    def _setup_embeddings(self):
        """Setup sentence transformer embedding model"""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        if self.verbose:
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded on {self.device}")
    
    def _setup_llm(self):
        """Setup LLM model with device configuration"""
        pipeline_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": 0
        }
        
        if self.device == "cuda":
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                task="text-generation",
                device_map="auto",
                model_kwargs={"torch_dtype": torch.float16},
                pipeline_kwargs=pipeline_kwargs
            )
        elif self.device == "mps":
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                task="text-generation",
                device=0,
                model_kwargs={"torch_dtype": torch.float32},
                pipeline_kwargs=pipeline_kwargs
            )
        else:  # CPU
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                task="text-generation",
                device=None,
                model_kwargs={"torch_dtype": torch.float32},
                pipeline_kwargs=pipeline_kwargs
            )
        
        if self.verbose:
            logger.info(f"LLM '{self.model_name}' loaded on {self.device}")
    
    def load_documents_from_files(self, file_paths: List[str]) -> List:
        """Load and process documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    if self.verbose:
                        logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {e}")
                continue
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        if self.verbose:
            logger.info(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks")
        
        return chunks
    
    def add_documents_to_vectorstore(self, documents: List) -> None:
        """Add documents to the vectorstore"""
        try:
            if not self.vectorstore:
                # Create new vectorstore
                self.vectorstore = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    index_name=self.index_name
                )
                if self.verbose:
                    logger.info(f"Created new vectorstore with {len(documents)} documents")
            else:
                # Add to existing vectorstore
                self.vectorstore.add_documents(documents)
                if self.verbose:
                    logger.info(f"Added {len(documents)} documents to existing vectorstore")
            
            self.has_documents = True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vectorstore: {e}")
            raise
    
    def setup_qa_chain(self) -> None:
        """Setup the question-answering chain"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Add documents first.")
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        if self.verbose:
            logger.info("QA chain setup complete")
    
    def initialize_with_documents(self, file_paths: List[str]) -> None:
        """
        Complete initialization with document loading
        
        Args:
            file_paths: List of file paths to load as knowledge base
        """
        if not self.is_initialized:
            raise RuntimeError("Core components not initialized")
        
        # Load documents
        documents = self.load_documents_from_files(file_paths)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        # Add to vectorstore
        self.add_documents_to_vectorstore(documents)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        if self.verbose:
            logger.info("RAG Chatbot Module fully initialized and ready for queries!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            
        Returns:
            Dict containing answer and source documents
        """
        if not self.qa_chain:
            return {
                "error": "System not ready. Please initialize with documents first.",
                "answer": None,
                "source_documents": None
            }
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "error": str(e),
                "answer": None,
                "source_documents": None
            }
    
    def add_more_documents(self, file_paths: List[str]) -> None:
        """
        Add more documents to existing knowledge base
        
        Args:
            file_paths: List of additional file paths to load
        """
        if not self.has_documents:
            raise RuntimeError("Initialize with documents first before adding more")
        
        documents = self.load_documents_from_files(file_paths)
        if documents:
            self.add_documents_to_vectorstore(documents)
    
    def is_ready(self) -> bool:
        """Check if the system is ready to answer queries"""
        return self.is_initialized and self.has_documents and self.qa_chain is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "initialized": self.is_initialized,
            "has_documents": self.has_documents,
            "ready": self.is_ready(),
            "device": self.device,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model_name,
            "index_name": self.index_name
        }


# Convenience function for quick setup
def create_rag_chatbot(file_paths: List[str], **kwargs) -> RAGChatbotModule:
    """
    Convenience function to create and initialize a RAG chatbot in one step
    
    Args:
        file_paths: List of document file paths
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized RAGChatbotModule instance
    """
    chatbot = RAGChatbotModule(**kwargs)
    chatbot.initialize_with_documents(file_paths)
    return chatbot


# Example usage as a script
def main():
    """Example usage when run as a script"""
    try:
        # Method 1: Step-by-step initialization
        chatbot = RAGChatbotModule(verbose=True)
        
        document_paths = [
            "/Users/pulkit/ProjectJal/backend/WHO.txt"
        ]
        
        chatbot.initialize_with_documents(document_paths)
        
        # Interactive chat loop
        print("\nRAG Chatbot Module is ready! Type 'quit' to exit.")
        while True:
            question = input("\nYou: ")
            if question.lower() == 'quit':
                break
            
            response = chatbot.query(question)
            
            if response["error"]:
                print(f"Error: {response['error']}")
            else:
                print(f"\nBot: {response['answer']}")
                print(f"\nSources used:")
                for i, doc in enumerate(response['source_documents']):
                    print(f"{i+1}. {doc.page_content[:100]}...")
    
    except Exception as e:
        logger.error(f"Failed to run chatbot: {e}")


if __name__ == "__main__":
    main()
