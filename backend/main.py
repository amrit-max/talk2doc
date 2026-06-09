import fitz
import os
import requests
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv(find_dotenv())


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API = os.getenv("OPENROUTER_API_KEY")
print("Loaded OPENROUTER_API_KEY:", "SET" if OPENROUTER_API else "MISSING")



extracted_chunks = []
index = None
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  



def embed_text(texts):
    """Convert list of text chunks into embeddings"""
    vectors = embedding_model.encode(texts)
    return np.array(vectors).astype("float32")

def create_vector_store(chunks):
    """Create FAISS index from chunks"""
    global index
    vectors = embed_text(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

def search_chunks(query, k=3):
    """Search most relevant chunks using FAISS"""
    if index is None:
        return []
    query_vec = embed_text([query])
    distances, positions = index.search(query_vec, k)
    result = [extracted_chunks[i] for i in positions[0]]
    return result



@app.post("/upload")
async def upload_file(file: UploadFile):
    global extracted_chunks, index

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    os.makedirs("temp", exist_ok=True)
    content = await file.read()
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        doc = fitz.open(file_path)
        full_text = "".join(page.get_text() for page in doc)
    except:
        raise HTTPException(status_code=500, detail="Failed to read PDF")


    extracted_chunks = [full_text[i:i+1200] for i in range(0, len(full_text), 1200)]


    create_vector_store(extracted_chunks)

    return {"message": "PDF uploaded & indexed successfully!", "chunks": len(extracted_chunks)}

@app.get("/ask")
async def ask(question: str):
    load_dotenv(find_dotenv())

    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API

    if not extracted_chunks:
        return {"answer": "Upload a document first."}

    if not api_key:
        return {"answer": "⚠️ OPENROUTER_API_KEY is not set. Please create/edit the backend/.env file with OPENROUTER_API_KEY=your_key and ask again."}

   
    context_chunks = search_chunks(question)
    combined_context = "\n\n".join(context_chunks)

    prompt = f"""
### CONTEXT ###
{combined_context}

### TASK ###
Answer the question based on the above context.

👉 If the question requires FACTS (like names, numbers, dates, address, rent amount), answer EXACTLY from the context.

👉 If the question requires ANALYSIS (like important keywords, summary, main points), analyze the context and generate the result.

🚫 If the required factual info is NOT present, reply EXACTLY: "Not available in document."

### QUESTION ###
{question}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    try:
        answer = response.json()["choices"][0]["message"]["content"]
        return {"answer": answer}
    except:
        return {"answer": "API Error: " + response.text}
