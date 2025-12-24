import requests, os
import xml.etree.ElementTree as ET

VULTR_ENDPOINT = "https://sgp1.vultrobjects.com"
BUCKET = "mabar-app"
PREFIX = "models/Wan2.2-T2V-A14B"

LOCAL_DIR = "./Wan2.2-T2V-A14B"

def list_vultr_objects():
    url = f"{VULTR_ENDPOINT}/{BUCKET}?prefix={PREFIX}/"
    print("[LIST]", url)
    r = requests.get(url)
    r.raise_for_status()

    root = ET.fromstring(r.text)

    files = []
    for content in root.findall(".//{http://s3.amazonaws.com/doc/2006-03-01/}Contents"):
        key = content.find("{http://s3.amazonaws.com/doc/2006-03-01/}Key").text
        if not key.endswith("/"):
            files.append(key)

    return files


def download_all():
    files = list_vultr_objects()

    print(f"[INFO] Found {len(files)} files")

    for key in files:
        rel = key.replace(PREFIX + "/", "")
        dst = os.path.join(LOCAL_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        url = f"{VULTR_ENDPOINT}/{BUCKET}/{key}"
        print("[DL]", url)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)

    print("[OK] All files downloaded")


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    download_all()
