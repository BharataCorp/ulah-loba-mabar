import boto3
import os

MODEL_BUCKET = os.environ["MODEL_S3_BUCKET"]
MODEL_ENDPOINT = os.environ["MODEL_S3_ENDPOINT"]
ACCESS_KEY = os.environ["MODEL_S3_ACCESS_KEY"]
SECRET_KEY = os.environ["MODEL_S3_SECRET_KEY"]

PREFIX = "models/Wan2.2-T2V-A14B"
DEST = "/workspace/Wan2.2/Wan2.2-T2V-A14B"

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{MODEL_ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

def download_all():
    paginator = s3.get_paginator("list_objects_v2")
    os.makedirs(DEST, exist_ok=True)

    for page in paginator.paginate(Bucket=MODEL_BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key.replace(PREFIX + "/", "")
            path = os.path.join(DEST, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            print("[DL]", key)
            s3.download_file(MODEL_BUCKET, key, path)

    print("ALL FILES DOWNLOADED")

download_all()
