import fitz
import pytesseract
from PIL import Image
import io

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\AmitkumarSingh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# PDF file path
pdf_path = "sample_docs/invoice.pdf"

# Open PDF
doc = fitz.open(pdf_path)

# Store extracted text
full_text = ""

# Process all pages
for page_num in range(len(doc)):

    # Load page
    page = doc.load_page(page_num)

    # Convert PDF page to image
    pix = page.get_pixmap()

    # Convert image bytes
    img_bytes = pix.tobytes("png")

    # Open image using PIL
    image = Image.open(io.BytesIO(img_bytes))

    # OCR text extraction
    text = pytesseract.image_to_string(image)

    # Append extracted text
    full_text += text

print("OCR Extraction Completed Successfully\n")

print("========== EXTRACTED TEXT ==========\n")

print(full_text[:3000])