# Talk2Doc AI — Premium AI Document Assistant

Talk2Doc is a full-stack web application that allows you to upload PDF documents and converse with them. It instantly analyzes, extracts facts, and answers questions based on your document using local FAISS embeddings and the OpenRouter API.

## 🚀 Features

- **PDF Processing**: Upload any PDF document to instantly extract text and knowledge chunks.
- **Local Embeddings**: Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to create high-quality vector embeddings locally.
- **Fast Retrieval**: Powered by **FAISS** (Facebook AI Similarity Search) to quickly find the most relevant chunks of text based on your query.
- **AI Conversation**: Integrates with the **OpenRouter API** (using `gpt-4o-mini`) to provide accurate, factual, and analytical answers derived strictly from the document context.
- **Modern UI**: A sleek, responsive, glassmorphism-inspired frontend built with Vanilla HTML/CSS/JS and Lucide Icons.

## 🛠️ Technology Stack

**Backend**
- Python 3
- FastAPI & Uvicorn
- PyMuPDF (fitz) for PDF text extraction
- Sentence-Transformers & FAISS for vector search
- Requests for API communication

**Frontend**
- HTML5, CSS3, Vanilla JavaScript
- Google Fonts (Plus Jakarta Sans)
- Lucide Icons

## 📋 Prerequisites

- Python 3.8 or higher
- An API Key from [OpenRouter](https://openrouter.ai/)

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/amrit-max/talk2doc.git
cd talk2doc
```

### 2. Backend Setup
Navigate to the `backend` directory and install the required Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file inside the `backend` folder and add your OpenRouter API Key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The backend server will start at `http://localhost:8000`.

### 3. Frontend Setup
The frontend is completely static and requires no build step. 
Simply open `frontend/index.html` in your web browser. 

*Tip: For the best experience, you can serve the frontend folder using a local web server (e.g., VS Code Live Server, Python's `http.server`, etc.).*

## 💡 How to Use
1. Ensure the FastAPI backend is running.
2. Open the frontend interface.
3. Click on the **Upload Dropzone** or drag and drop a PDF file.
4. Click **Index Document** to process the PDF and create the FAISS index.
5. Once indexed, start asking questions in the chat console! The AI will respond based on the contents of the uploaded PDF.

## 📄 License
This project is open-source and available under the MIT License.
