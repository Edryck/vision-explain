import base64
import io
import json
import sys
import zipfile

from src.multicategory import EXPERIMENT_DIR


def find_payload(notebook_path):
    with open(notebook_path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if "EXP007_ZIP_BEGIN" in text:
                return text.split("EXP007_ZIP_BEGIN", 1)[1].split("EXP007_ZIP_END", 1)[0].strip()
    return None


if __name__ == "__main__":
    payload = find_payload(sys.argv[1])
    if payload is None:
        sys.exit("Sem resultados no notebook: rode a célula de exportação e salve o .ipynb.")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        archive.extractall(EXPERIMENT_DIR)
        for name in archive.namelist():
            print(EXPERIMENT_DIR / name)
