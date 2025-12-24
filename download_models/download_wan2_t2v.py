import boto3
import os
import requests

# =========================
# CONFIG
# =========================
BUCKET_NAME = "mabar-app"
PREFIX = "models/Wan2.2-T2V-A14B"
ENDPOINT = "https://sgp1.vultrobjects.com"

# =========================
# INIT S3 CLIENT (PUBLIC)
# =========================
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id="",
    aws_secret_access_key="",
)

# =========================
# LIST ALL OBJECTS
# =========================
print(f"Listing objects in s3://{BUCKET_NAME}/{PREFIX} ...")

objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)

if "Contents" not in objects:
    print("No objects found. Check prefix or permissions.")
    exit()

# =========================
# DOWNLOAD ALL FILES
# =========================
for obj in objects["Contents"]:
    key = obj["Key"]
    if key.endswith("/"):  # skip directory markers
        continue

    relative_path = key[len(PREFIX) + 1:]  # remove prefix from path

    local_path = f"./workspace/Wan2.2/Wan2.2-T2V-A14B/{relative_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    file_url = f"{ENDPOINT}/{BUCKET_NAME}/{key}"

    print(f"Downloading: {file_url} → {local_path}")

    # stream download
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("\nDone! All files downloaded successfully.")
