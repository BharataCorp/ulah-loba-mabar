import os
import hashlib
from collections import defaultdict

ROOTS = [
    "/workspace/Wan2.2/Wan2.2-Animate-14B",
    "/workspace/Wan2.2/Wan2.2-I2V-A14B",
    "/workspace/Wan2.2/Wan2.2-S2V-14B",
    "/workspace/Wan2.2/Wan2.2-T2V-A14B",
    "/workspace/Wan2.2/Wan2.2-TI2V-5B",
]

hash_map = defaultdict(list)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for root in ROOTS:
    for dirpath, dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)

            if os.path.getsize(path) < 100 * 1024 * 1024:
                continue

            try:
                h = sha256(path)
                hash_map[h].append(path)
            except:
                pass

for h, files in hash_map.items():
    if len(files) > 1:
        print("\n[DUPLICATE MODEL]")
        for f in files:
            print(" ", f)
