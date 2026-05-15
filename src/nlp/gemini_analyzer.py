from google import genai
from dotenv import load_dotenv
import os
import fitz
import pytesseract
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AmitkumarSingh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# PDF file path
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

# Gemini business analysis prompt
prompt = f"""
You are an AI fashion business analyst.

Analyze this document and provide:

1. Executive Summary
2. Important Business Insights
3. Pricing Analysis
4. Inventory Analysis
5. Supplier Observations
6. Risks or anomalies
7. Strategic Recommendations

Document Content:
{full_text}
"""

# Gemini analysis
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n========== GEMINI BUSINESS ANALYSIS ==========\n")

print(response.text)