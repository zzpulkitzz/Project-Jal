 AquaBot - Water Conservation RAG Chatbot

## 📋 Problem Statement

A chatbot to educate communities about water conservation techniques and best practices for maintaining clean water. The chatbot answers questions about water-saving methods, provides tips for reducing water usage, and raises awareness about the importance of sanitation.
    
## ✨ Features

### 🤖 RAG Chatbot Capabilities
- **Intelligent Q&A**: Answers questions about water conservation using retrieval-augmented generation
- **Document Processing**: Supports PDF and TXT files for knowledge base creation
- **Source Attribution**: Provides source documents for transparency and verification
- **Contextual Understanding**: Maintains conversation context for natural interactions

### 🔧 Backend Features
- **Modular Architecture**: Reusable RAG chatbot module for easy integration
- **FastAPI REST API**: Production-ready backend with comprehensive endpoints
- **Multiple Device Support**: Automatic detection and optimization for CPU, CUDA, and Apple Silicon (MPS)
- **Document Management**: Add, upload, and manage knowledge base documents
- **Health Monitoring**: System status and health check endpoints
- **Error Handling**: Comprehensive error handling with user-friendly messages

### 🎨 Frontend Features
- **Water-Themed UI**: Beautiful design with water conservation aesthetics
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Chat**: Smooth chat interface with typing indicators
- **Quick Questions**: Pre-defined questions for easy interaction
- **File Upload**: Direct document upload to expand knowledge base
- **Status Indicators**: Real-time system status and connectivity feedback

### 🧠 AI/ML Components
- **LLM**: TinyLlama-1.1B-Chat-v1.0 for text generation
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 for semantic search
- **Vector Database**: Pinecone for efficient similarity search
- **Framework**: LangChain for RAG orchestration

## 🏗️ Project Structure

water-conservation-chatbot/
├── backend/
│ ├── main.py # FastAPI entry point
│ ├── rag_chatbot_module.py # Core RAG chatbot class
│ ├── requirements.txt # Python dependencies
│ └── .env # Environment variables
├── frontend/
│ ├── src/
│ │ ├── App.jsx # Main React component
│ │ ├── index.css # Tailwind styles
│ │ └── main.jsx # React entry point
│ ├── package.json # Node.js dependencies
│ ├── tailwind.config.js # Tailwind configuration
│ ├── postcss.config.js # PostCSS configuration
│ └── vite.config.js # Vite configuration
├── documents/ # Knowledge base documents
│ └── water-conservation-guide.txt
└── README.md


## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Pinecone Account** (free tier available)
- **Git**

### 1. Clone the Repository

git clone <repository-url>
cd water-conservation-chatbot

text

### 2. Backend Setup

#### Environment Variables

Create a `.env` file in the `backend/` directory:

PINECONE_API_KEY=your_pinecone_api_key_here

text

#### Install Dependencies

cd backend
pip install -r requirements.txt

text

#### Requirements.txt

fastapi[standard]
uvicorn[standard]
python-dotenv
langchain
langchain-huggingface
langchain-pinecone
langchain-text-splitters
pinecone
pypdf
transformers
sentence-transformers
torch
accelerate
python-multipart

text

#### Start Backend Server

Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

text

### 3. Frontend Setup

#### Install Dependencies

cd frontend
npm install

text

#### Package Dependencies

The frontend uses:
- **React 18+** with Vite
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Axios** for API calls

#### Start Development Server

npm run dev

text

The frontend will be available at `http://localhost:5173`

### 4. Pinecone Setup

1. **Create Account**: Sign up at [pinecone.io](https://pinecone.io)
2. **Get API Key**: Copy your API key from the dashboard
3. **Index Creation**: The application automatically creates the required index

## 📚 API Documentation

Once the backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/initialize` | POST | Initialize knowledge base with documents |
| `/query` | POST | Ask questions to the chatbot |
| `/upload-documents` | POST | Upload and add documents |
| `/status` | GET | Get system status |
| `/health` | GET | Health check |

### Example API Usage

Initialize with documents
curl -X POST "http://localhost:8000/initialize"
-H "Content-Type: application/json"
-d '{"file_paths": ["/path/to/water-guide.txt"]}'

Query the chatbot
curl -X POST "http://localhost:8000/query"
-H "Content-Type: application/json"
-d '{"question": "How can I save water at home?"}'

text

## 🔧 Configuration

### Backend Configuration

Modify `rag_chatbot_module.py` for:

chatbot = RAGChatbotModule(
index_name="water-conservation-kb", # Pinecone index name
model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # LLM model
chunk_size=1000, # Document chunk size
chunk_overlap=200, # Chunk overlap
retrieval_k=3, # Number of retrieved documents
max_tokens=512, # Max generation tokens
temperature=0.7 # Generation temperature
)

text

### Frontend Configuration

Update API URL in `App.jsx`:

const API_BASE_URL = 'http://your-backend-url:8000';

text

## 🎯 Usage Examples

### 1. Initialize Knowledge Base

const response = await fetch('/initialize', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
file_paths: ['/path/to/water-conservation-guide.txt']
})
});

text

### 2. Query the Chatbot

const response = await fetch('/query', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
question: 'What are the best water conservation techniques?'
})
});

text

### 3. Upload Documents

const formData = new FormData();
formData.append('files', file);

const response = await fetch('/upload-documents', {
method: 'POST',
body: formData
});

text

## 🐛 Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error
Solution: Downgrade NumPy
pip uninstall numpy
pip install "numpy<2.0"

text

#### 2. MPS Tensor Error (macOS)
The application automatically handles this by falling back to CPU
For macOS 14.0+, MPS should work properly
text

#### 3. Tailwind CSS PostCSS Error
Install correct PostCSS plugin
npm install @tailwindcss/postcss

text

#### 4. Pinecone Connection Issues
- Verify API key in `.env` file
- Check Pinecone dashboard for quota limits
- Ensure network connectivity

### Backend Logs

Check logs for debugging:
View logs
tail -f /var/log/aquabot.log

Debug mode
uvicorn main:app --log-level debug

text

## 🚀 Deployment

### Backend Deployment

#### Docker (Recommended)

FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

text

#### Cloud Platforms
- **AWS**: Use ECS, Lambda, or EC2
- **Google Cloud**: Cloud Run or Compute Engine
- **Azure**: Container Instances or App Service
- **Railway/Render**: Simple deployment options

### Frontend Deployment

Build for production
npm run build

Deploy to Vercel, Netlify, or any static hosting
text

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **Pinecone** for vector database
- **Hugging Face** for models and embeddings
- **FastAPI** for backend framework
- **React & Tailwind** for frontend
- **OpenAI** for inspiration in conversational AI



---
