import streamlit as st
import fitz
import faiss
import numpy as np
import tempfile
import os
import io
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit config
st.set_page_config(
    page_title="Fashion AI Document Intelligence",
    layout="wide"
)

# Title
st.title("Fashion AI Document Intelligence Platform")

# Sidebar
st.sidebar.title("AI Business Copilot")

# Upload section
uploaded_file = st.file_uploader(
    "Upload Fashion Business Document",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully")

    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Open PDF
    doc = fitz.open(pdf_path)

    full_text = ""

    # STEP 1 → Direct text extraction
    with st.spinner("Extracting Text From PDF..."):

        for page in doc:
            text = page.get_text()

            if text.strip():
                full_text += text

    # STEP 2 → OCR fallback
    if full_text.strip() == "":

        st.warning("Scanned PDF detected. Using OCR extraction...")

        for page in doc:

            pix = page.get_pixmap()

            img_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(image)

            full_text += text

    st.success("Text Extraction Completed")

    # Create chunks
    chunks = full_text.split("\n")

    chunks = [chunk for chunk in chunks if chunk.strip() != ""]

    st.info(f"Total Text Chunks: {len(chunks)}")

    # Prevent empty PDF crash
    if len(chunks) == 0:
        st.error("No readable text found in PDF.")
        st.stop()

    # Generate embeddings
    with st.spinner("Generating Embeddings..."):

        embeddings = embedding_model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")

    # Create vector DB
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success("Vector Database Created")

    # Question section
    st.subheader("Ask Questions About Document")

    user_question = st.text_input(
        "Enter your business question"
    )

    if user_question:

        with st.spinner("Analyzing Document..."):

            # Query embedding
            query_embedding = embedding_model.encode([user_question])

            query_embedding = np.array(query_embedding).astype("float32")

            # Search
            k = 5

            distances, indices = index.search(query_embedding, k)

            # Context retrieval
            retrieved_context = ""

            for idx in indices[0]:
                retrieved_context += chunks[idx] + "\n"

            # Prompt
            prompt = f"""
            You are an AI fashion business analyst.

            Answer ONLY from document context.

            Document Context:
            {retrieved_context}

            User Question:
            {user_question}
            """

            # Gemini response
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        # Show answer
        st.subheader("AI Business Insight")

        st.write(response.text)