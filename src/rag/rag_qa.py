import fitz
import pytesseract
from PIL import Image
import io
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AmitkumarSingh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding Model Loaded")

# PDF path
pdf_path = "sample_docs/invoice.pdf"

# Open PDF
doc = fitz.open(pdf_path)

# Extract OCR text
full_text = ""

for page_num in range(len(doc)):

    page = doc.load_page(page_num)

    pix = page.get_pixmap()

    img_bytes = pix.tobytes("png")

    image = Image.open(io.BytesIO(img_bytes))

    text = pytesseract.image_to_string(image)

    full_text += text

print("OCR Extraction Completed")

# Create chunks
chunks = full_text.split("\n")

chunks = [chunk for chunk in chunks if chunk.strip() != ""]

print(f"Total Chunks: {len(chunks)}")

# Generate embeddings
embeddings = embedding_model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS Vector Store Ready")

# Continuous AI chat loop
while True:

    # User input
    query = input("\nAsk Question About Document (type 'exit' to quit): ")

    # Exit condition
    if query.lower() == "exit":
        print("AI Assistant Closed")
        break

    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Semantic search
    k = 5

    distances, indices = index.search(query_embedding, k)

    # Retrieve context
    retrieved_context = ""

    for idx in indices[0]:
        retrieved_context += chunks[idx] + "\n"

    print("\nRetrieved Context Ready")

    # Gemini RAG prompt
    prompt = f"""
    You are an AI fashion business analyst.

    Answer the user question ONLY using the retrieved document context.

    Retrieved Context:
    {retrieved_context}

    User Question:
    {query}
    """

    # Gemini response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\n========== AI ANSWER ==========\n")

    print(response.text)