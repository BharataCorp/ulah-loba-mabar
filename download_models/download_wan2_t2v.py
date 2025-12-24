import requests
import os
import xml.etree.ElementTree as ET

ENDPOINT = "https://sgp1.vultrobjects.com"
BUCKET = "mabar-app"
PREFIX = "models/Wan2.2-T2V-A14B"
DEST = "./Wan2.2-T2V-A14B"

def list_objects():
    url = f"{ENDPOINT}/{BUCKET}?prefix={PREFIX}/"
    print("[LIST]", url)
    r = requests.get(url)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    files = []
    for c in root.findall("s3:Contents", ns):
        key = c.find("s3:Key", ns).text
        if not key.endswith("/"):
            files.append(key)

    return files

def download_all():
    os.makedirs(DEST, exist_ok=True)
    files = list_objects()

    print(f"[INFO] Found {len(files)} files")

    for key in files:
        rel = key.replace(PREFIX + "/", "")
        path = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        url = f"{ENDPOINT}/{BUCKET}/{key}"
        print("[DL]", url)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)

    print("DONE")

if __name__ == "__main__":
    download_all()
