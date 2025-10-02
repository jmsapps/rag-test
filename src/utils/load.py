import pathlib

from typing import Dict


def load_docs_from_folder(folder_path) -> list[Dict[str, str]]:
    """Load all .txt files from docs/ folder."""
    docs = []
    for f in pathlib.Path(folder_path).glob("*.txt"):
        with open(f, "r", encoding="utf-8") as fh:
            docs.append({"title": f.stem, "content": fh.read()})
    print(f"Loaded {len(docs)} document(s) from '{folder_path}'.")

    return docs
