import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Local file path
file_path = "sample_docs/invoice.pdf"

# File name
file_name = os.path.basename(file_path)

# Upload to S3
s3.upload_file(
    file_path,
    os.getenv("S3_BUCKET_NAME"),
    f"raw-data/pdfs/{file_name}"
)

print(f"{file_name} uploaded successfully")