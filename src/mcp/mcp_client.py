import requests
import json
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MCP endpoint
MCP_URL = "https://dummyjson.com/products"

# Fetch data
response = requests.get(MCP_URL)

# Convert response to JSON
data = response.json()

print("MCP Data Fetched Successfully")
print(f"Total Products: {len(data['products'])}")

print("\nSample Product:")
print(data['products'][0]['title'])

# Save locally
with open("products.json", "w") as file:
    json.dump(data, file, indent=4)

print("JSON Saved Locally")

# Create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Upload to S3
s3.upload_file(
    "products.json",
    os.getenv("S3_BUCKET_NAME"),
    "raw-data/mcp/products.json"
)

print("JSON Uploaded to S3 Successfully")