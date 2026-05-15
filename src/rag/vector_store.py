import fitz
import pytesseract
from PIL import Image
import io
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("STEP 1: Imports successful")

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AmitkumarSingh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

print("STEP 2: Tesseract configured")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("STEP 3: Embedding model loaded")

# PDF path
pdf_path = "sample_docs/invoice.pdf"

# Open PDF
doc = fitz.open(pdf_path)

print("STEP 4: PDF opened")

# Extract OCR text
full_text = ""

for page_num in range(len(doc)):

    page = doc.load_page(page_num)

    pix = page.get_pixmap()

    img_bytes = pix.tobytes("png")

    image = Image.open(io.BytesIO(img_bytes))

    text = pytesseract.image_to_string(image)

    full_text += text

print("STEP 5: OCR completed")

# Split text into chunks
chunks = full_text.split("\n")

# Remove empty chunks
chunks = [chunk for chunk in chunks if chunk.strip() != ""]

print(f"STEP 6: Total chunks = {len(chunks)}")

# Generate embeddings
embeddings = embedding_model.encode(chunks)

print("STEP 7: Embeddings generated")

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

print("STEP 8: FAISS index created")

# Semantic search query
query = "supplier pricing analysis"

query_embedding = embedding_model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

# Search
k = 3

distances, indices = index.search(query_embedding, k)

print("\n========== SEARCH RESULTS ==========\n")

for idx in indices[0]:
    print(chunks[idx])
    print("\n-----------------\n")