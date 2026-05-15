import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io
import faiss
import numpy as np
import tempfile
import os

from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AmitkumarSingh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit page config
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

# Process uploaded PDF
if uploaded_file is not None:

    st.success("PDF Uploaded Successfully")

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Open PDF
    doc = fitz.open(pdf_path)

    # OCR extraction
    full_text = ""

    with st.spinner("Performing OCR Extraction..."):

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            pix = page.get_pixmap()

            img_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(image)

            full_text += text

    st.success("OCR Extraction Completed")

    # Create chunks
    chunks = full_text.split("\n")

    chunks = [chunk for chunk in chunks if chunk.strip() != ""]

    st.info(f"Total Text Chunks: {len(chunks)}")

    # Generate embeddings
    with st.spinner("Generating Embeddings..."):

        embeddings = embedding_model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")

    # Create FAISS vector DB
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success("Vector Database Created")

    # Ask question section
    st.subheader("Ask Questions About Document")

    user_question = st.text_input(
        "Enter your business question"
    )

    # AI question answering
    if user_question:

        with st.spinner("Analyzing Document..."):

            # Convert query to embedding
            query_embedding = embedding_model.encode([user_question])

            query_embedding = np.array(query_embedding).astype("float32")

            # Semantic search
            k = 5

            distances, indices = index.search(query_embedding, k)

            # Retrieve relevant context
            retrieved_context = ""

            for idx in indices[0]:
                retrieved_context += chunks[idx] + "\n"

            # Gemini prompt
            prompt = f"""
            You are an AI fashion business analyst.

            Answer the question ONLY using the document context.

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

        # Display answer
        st.subheader("AI Business Insight")

        st.write(response.text)