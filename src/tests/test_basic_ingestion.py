import time
import json
import pathlib
import requests
import tiktoken

from models.azure import AzureOpenAIModel


def delete_index_if_exists(config):
    """Delete index if it exists and wait until confirmed deleted."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }

    r = requests.delete(url, headers=headers)
    if r.status_code in (200, 204):
        print(
            f"🗑️ Deleted existing index '{config['AZURE_SEARCH_API_INDEX']}'. Waiting for confirmation..."
        )
        for _ in range(10):
            check = requests.get(url, headers=headers)
            if check.status_code == 404:
                print("✅ Index deletion confirmed.")
                return
            time.sleep(1)
        print("⚠️ Index still exists after 10s — proceeding anyway.")
    elif r.status_code == 404:
        print("ℹ️ No existing index to delete.")
    else:
        raise RuntimeError(f"Could not delete index: {r.status_code}, {r.text}")


def create_hybrid_index(config, embedding_dim=1536):
    """Create a fresh hybrid index with vector + keyword search fields, compatible with API 2023-11-01+."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }

    index_schema = {
        "name": config["AZURE_SEARCH_API_INDEX"],
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "sortable": True,
            },
            {"name": "content", "type": "Edm.String", "searchable": True},
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": True,
                "filterable": False,
                "facetable": False,
                "sortable": False,
                "dimensions": embedding_dim,
                "vectorSearchProfile": "default-hnsw-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "default-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "metric": "cosine"
                        # optionally other params like efConstructions, etc.
                    },
                }
            ],
            "profiles": [{"name": "default-hnsw-profile", "algorithm": "default-hnsw"}],
        },
    }

    r = requests.put(url, headers=headers, data=json.dumps(index_schema))
    if r.status_code >= 400:
        raise RuntimeError(f"Index creation failed: {r.status_code}, {r.text}")
    print(
        f"✅ Created hybrid vector+keyword index '{config['AZURE_SEARCH_API_INDEX']}'."
    )


def load_docs_from_folder(folder_path):
    """Load all .txt files from docs/ folder."""
    docs = []
    for f in pathlib.Path(folder_path).glob("*.txt"):
        with open(f, "r", encoding="utf-8") as fh:
            docs.append({"title": f.stem, "content": fh.read()})
    print(f"📄 Loaded {len(docs)} document(s) from '{folder_path}'.")
    return docs


def chunk_text(text, max_tokens=500, overlap=50):
    """Split text into overlapping chunks using token count."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks


def ingest_docs(config, docs):
    """Chunk, embed, and upload documents to Azure Cognitive Search."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/index?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }
    upload_payload = []

    for doc_id, doc in enumerate(docs, start=1):
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            embedding = AzureOpenAIModel.azure_openai_generate_embedding(chunk)

            print(f"Created embedding {embedding}")
            upload_payload.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": f"{doc_id}-{i}",
                    "title": doc["title"],
                    "content": chunk,
                    "embedding": embedding,
                }
            )

    r = requests.post(url, headers=headers, data=json.dumps({"value": upload_payload}))
    if r.status_code >= 400:
        raise RuntimeError(f"Ingestion failed: {r.status_code}, {r.text}")
    print(f"✅ Ingested {len(upload_payload)} chunks with embeddings.")


def main(config):
    delete_index_if_exists(config)
    create_hybrid_index(config)

    docs = load_docs_from_folder("src/docs")
    if not docs:
        raise RuntimeError(
            "No documents found in ./docs — please add at least one .txt file."
        )

    ingest_docs(config, docs)
