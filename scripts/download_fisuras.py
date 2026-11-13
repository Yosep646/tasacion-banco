"""Download checksum-pinned OpenSistemas YOLOv8n crack-seg model from Hugging Face.

WARNING: .pt is a pickle-backed format. Only run with a trusted model and after
reviewing its origin, license (AGPL-3.0) and security implications.
"""
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlopen

URL = ("https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/"
       "2d6edb106ef9681ead363e5189c68dabeb27f5ef/yolov8n/weights/best.pt")
SHA256 = "cce8b30758621622a03119677709d172f2ec1a2f0452279bda37732fcffb00af"


def download(destination="weights/vivienda-fisuras.pt"):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    try:
        digest = hashlib.sha256()
        with urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            while True:
                piece = response.read(1024 * 1024)
                if not piece:
                    break
                digest.update(piece)
                output.write(piece)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("SHA256 no coincide: archivo de pesos rechazado")
        partial.replace(path)
        return path
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    print("Peso verificado:", download(sys.argv[1] if len(sys.argv)>1 else "weights/vivienda-fisuras.pt"))
