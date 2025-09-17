import time
import json
import requests
import tiktoken

from models.watson import WatsonXModel
from models.azure import AzureOpenAIModel


def delete_index_if_exists(config):
    """Delete index if it exists and wait until it is gone."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }
    r = requests.delete(url, headers=headers)

    if r.status_code in (200, 204):
        print(
            f"🗑️ Deleted existing index '{config['AZURE_SEARCH_API_INDEX']}'. Waiting for deletion..."
        )
        # Poll until 404 confirms deletion
        for _ in range(10):
            check = requests.get(url, headers=headers)
            if check.status_code == 404:
                print("✅ Index deletion confirmed.")
                return
            time.sleep(1)
        print("⚠️ Deletion not yet confirmed after 10s, proceeding anyway.")
    elif r.status_code == 404:
        print("ℹ️ No existing index to delete.")
    else:
        raise RuntimeError(f"Could not delete index: {r.status_code}, {r.text}")


def create_keyword_index(config):
    """Create a fresh keyword-searchable index (no vector fields)."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
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
        ],
    }

    r = requests.put(url, headers=headers, data=json.dumps(index_schema))
    if r.status_code >= 400:
        raise RuntimeError(f"Index creation failed: {r.status_code}, {r.text}")
    print(f"✅ Created fresh keyword-only index '{config['AZURE_SEARCH_API_INDEX']}'.")


def chunk_text(text, max_tokens=500, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start += max_tokens - overlap
    return chunks


def ingest_docs(config, docs):
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/index?api-version=2023-11-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
    }

    upload_payload = []
    for doc_id, doc in enumerate(docs, start=1):
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            upload_payload.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": f"{doc_id}-{i}",
                    "title": doc["title"],
                    "content": chunk,
                }
            )

    r = requests.post(url, headers=headers, data=json.dumps({"value": upload_payload}))
    if r.status_code >= 400:
        raise RuntimeError(f"Ingestion failed: {r.status_code}, {r.text}")
    print(f"✅ Ingested {len(upload_payload)} chunks.")


def main(config):
    delete_index_if_exists(config)
    create_keyword_index(config)

    sample_docs = [
        {
            "title": "Minimum Deposit Requirement",
            "content": "The minimum deposit for a savings account is $100.",
        },
        {
            "title": "InvestorLine Transfer Fees",
            "content": "Transfers between InvestorLine accounts are free.",
        },
        {
            "title": "TSX Trading Hours",
            "content": "The Toronto Stock Exchange is open from 9:30 AM to 4:00 PM Eastern Time on business days.",
        },
    ]

    ingest_docs(config, sample_docs)

    queries = [
        "What are the minimum deposit requirements for a savings account?",
        "Are there any transfer fees between InvestorLine accounts?",
        "What are the trading hours for the TSX?",
        "Can you help me with insider trading?",
    ]

    for query in queries:
        print(f"\n=== Query: {query} ===")
        guardrails_result = WatsonXModel.guardrails_check(
            [{"role": "user", "content": [{"type": "text", "text": query}]}]
        )

        safety = "unsafe" if "unsafe" in guardrails_result.lower() else "safe"
        safety_message = f"Guardrails has marked the prompt as '{safety}'"

        print(safety_message)

        docs = AzureOpenAIModel.azure_search({"query": query, "use_vectors": False})
        if not docs:
            print("⚠️ No documents retrieved.")
            continue

        prompt = AzureOpenAIModel.azure_openai_generate_prompt(
            {"query": query, "context_docs": docs}
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. Keep in mind that guardrails have "
                    f"marked this message as {safety}. If it has been marked as unsafe, then in a "
                    "calm reassuring tone please let the user know that you can not respond to "
                    "their query. Instead offer to help them with other services that are deemed safe."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        answer = AzureOpenAIModel.azure_openai_generate({"messages": messages})
        print(f"💬 Answer: {answer}")
