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

# Test S3 connection
response = s3.list_buckets()

print("S3 Connected Successfully")

for bucket in response["Buckets"]:
    print(bucket["Name"])