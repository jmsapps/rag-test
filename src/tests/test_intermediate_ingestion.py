# test_intermediate_ingestion.py
import re
import json
import time
import pathlib
import requests
import tiktoken
from typing import List, Dict

from models.azure import AzureOpenAIModel


# --------------------------
# Index management
# --------------------------


def delete_index_if_exists(config):
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }
    r = requests.delete(url, headers=headers)
    if r.status_code in (200, 204):
        print(
            f"Deleted '{config['AZURE_SEARCH_API_INDEX']}'. Waiting for confirmation..."
        )
        for _ in range(10):
            if requests.get(url, headers=headers).status_code == 404:
                print("Index deletion confirmed.")
                return
            time.sleep(1)
        print("Index still exists after 10s — proceeding anyway.")
    elif r.status_code == 404:
        print("No existing index to delete.")
    else:
        raise RuntimeError(f"Could not delete index: {r.status_code}, {r.text}")


def create_hybrid_index(config, embedding_dim=1536):
    """Fields:
    - chunk_type: 'heading' | 'summary' | 'detailed'
    - content: holds the text for ALL types
    """
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }
    index_schema = {
        "name": config["AZURE_SEARCH_API_INDEX"],
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            {"name": "doc_id", "type": "Edm.String", "filterable": True},
            {"name": "section_id", "type": "Edm.String", "filterable": True},
            {
                "name": "chunk_type",
                "type": "Edm.String",
                "filterable": True,
            },  # heading|summary|detailed
            {
                "name": "order",
                "type": "Edm.Int32",
                "sortable": True,
                "filterable": True,
            },
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "sortable": True,
            },
            {"name": "section_heading", "type": "Edm.String", "searchable": True},
            # Single text field used for all types (heading/summary/detailed)
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
                    "hnswParameters": {"metric": "cosine"},
                }
            ],
            "profiles": [{"name": "default-hnsw-profile", "algorithm": "default-hnsw"}],
        },
    }
    r = requests.put(url, headers=headers, data=json.dumps(index_schema))
    if r.status_code >= 400:
        raise RuntimeError(f"Index creation failed: {r.status_code}, {r.text}")
    print(f"Created hybrid index '{config['AZURE_SEARCH_API_INDEX']}'.")


# --------------------------
# Loading + splitting
# --------------------------


def load_docs_from_folder(folder_path: str) -> List[Dict[str, str]]:
    docs = []
    for f in pathlib.Path(folder_path).glob("*.txt"):
        with open(f, "r", encoding="utf-8") as fh:
            docs.append({"title": f.stem, "content": fh.read()})
    print(f"Loaded {len(docs)} document(s) from '{folder_path}'.")
    return docs


def _tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def chunk_by_tokens(text: str, max_tokens=600, overlap=60) -> List[str]:
    tok = _tokenizer()
    ids = tok.encode(text)
    chunks, start = [], 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(tok.decode(ids[start:end]))
        start += step
    return chunks


# --------------------------
# UTILS
# --------------------------


def _extract_json(s: str) -> str:
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found")
    return s[first : last + 1]


def heuristic_section_document(text: str) -> List[Dict[str, str]]:
    HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
    matches = list(HEADING_RE.finditer(text))
    if matches:
        parts = []
        for i, m in enumerate(matches):
            if i > 0:
                prev = matches[i - 1]
                parts.append((text[prev.start() : m.start()], prev.group(2).strip()))
            if i == len(matches) - 1:
                parts.append((text[m.start() :], m.group(2).strip()))
        sections = []
        for body, heading in parts:
            body_wo = HEADING_RE.sub("", body, count=1).strip()
            if body_wo:
                sections.append({"heading": heading, "content": body_wo})
        if sections:
            return sections

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sections, buf, buf_len = [], [], 0
    for p in paras:
        buf.append(p)
        buf_len += len(p)
        if buf_len > 2500:
            sections.append({"heading": "", "content": "\n\n".join(buf)})
            buf, buf_len = [], 0
    if buf:
        sections.append({"heading": "", "content": "\n\n".join(buf)})
    if not sections:
        sections = [{"heading": "", "content": text}]
    return sections


def best_effort_heading(raw_heading: str, content: str, fallback: str) -> str:
    if raw_heading and raw_heading.strip():
        return raw_heading.strip()
    # fallback: first non-empty line or provided default
    first_line = (content.strip().splitlines() or [""])[0].strip()
    return first_line[:120] if first_line else fallback


# --------------------------
# LLM helpers
# --------------------------


def llm_section_document(text: str) -> List[Dict[str, str]]:
    """Ask the LLM for a JSON outline of sections. Fallback to heuristic on failure."""
    messages = [
        {
            "role": "system",
            "content": (
                "You split documents into coherent sections. "
                "Return STRICT JSON only:\n"
                '{ "sections": [ { "heading": "string", "content": "string" }, ... ] } '
                "Keep sections <= ~2000 tokens and do not omit important content."
            ),
        },
        {"role": "user", "content": text},
    ]
    raw = AzureOpenAIModel.azure_openai_generate({"messages": messages})
    try:
        js = json.loads(_extract_json(raw))
        sections = js.get("sections", [])
        ok = [s for s in sections if isinstance(s, dict) and "content" in s]
        if ok:
            return [
                {"heading": (s.get("heading") or "").strip(), "content": s["content"]}
                for s in ok
            ]
    except Exception:
        pass
    return heuristic_section_document(text)


def llm_summarize_text(text: str) -> str:
    """Return a 1–2 sentence summary (plain text)."""
    messages = [
        {
            "role": "system",
            "content": "Summarize the passage in 1–2 concise sentences. Output only the summary.",
        },
        {"role": "user", "content": text},
    ]
    resp = AzureOpenAIModel.azure_openai_generate({"messages": messages})
    return resp.strip()


# --------------------------
# Ingestion
# --------------------------


def ingest_docs(config, docs):
    """
    For each section emit:
      - heading row:   chunk_type='heading',  content = section heading text
      - summary row:   chunk_type='summary',  content = LLM summary of section
      - detailed rows: chunk_type='detailed', content = token-chunked section body
    Also emits a doc-level summary (chunk_type='summary', section_id="").
    """
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/index?api-version=2023-11-01"
    headers = {
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        "Content-Type": "application/json",
    }
    payload = []

    for doc_num, doc in enumerate(docs, start=1):
        print(f"Generating embedding for document '{doc_num}'")

        doc_id = str(doc_num)
        title = doc["title"]
        full_text = doc["content"]

        print(f"Sectioning doc {doc_id} with LLM")
        sections = llm_section_document(full_text)

        # Doc-level summary (coarse)
        print("Summarizing text with LLM")
        doc_summary_text = (
            llm_summarize_text(full_text) if len(full_text) > 400 else full_text
        )
        payload.append(
            {
                "@search.action": "mergeOrUpload",
                "id": f"{doc_id}-docsum",
                "doc_id": doc_id,
                "section_id": "",
                "chunk_type": "summary",
                "order": 0,
                "title": title,
                "section_heading": "Document",
                "content": doc_summary_text,
                "embedding": AzureOpenAIModel.azure_openai_generate_embedding(
                    doc_summary_text
                ),
            }
        )

        # Sections → heading + summary + detailed
        for s_idx, section in enumerate(sections, start=1):
            section_id = f"{doc_id}-s{s_idx}"
            section_heading = best_effort_heading(
                section.get("heading") or "",
                section.get("content", ""),
                f"Section {s_idx}",
            )
            section_body = section.get("content", "")

            # heading row
            heading_text = section_heading
            payload.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": f"{section_id}-h",
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "chunk_type": "heading",
                    "order": 0,
                    "title": title,
                    "section_heading": section_heading,
                    "content": heading_text,
                    "embedding": AzureOpenAIModel.azure_openai_generate_embedding(
                        heading_text
                    ),
                }
            )

            # summary row (per-section)
            section_summary = llm_summarize_text(section_body) if section_body else ""
            payload.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": f"{section_id}-sum",
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "chunk_type": "summary",
                    "order": 1,
                    "title": title,
                    "section_heading": section_heading,
                    "content": section_summary,
                    "embedding": AzureOpenAIModel.azure_openai_generate_embedding(
                        section_summary or section_heading
                    ),
                }
            )

            # detailed rows
            chunks = (
                chunk_by_tokens(section_body, max_tokens=600, overlap=60)
                if section_body
                else []
            )
            for c_idx, chunk in enumerate(chunks, start=1):
                payload.append(
                    {
                        "@search.action": "mergeOrUpload",
                        "id": f"{section_id}-d{c_idx}",
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "chunk_type": "detailed",
                        "order": 2 + c_idx,
                        "title": title,
                        "section_heading": section_heading,
                        "content": chunk,
                        "embedding": AzureOpenAIModel.azure_openai_generate_embedding(
                            chunk
                        ),
                    }
                )

    r = requests.post(url, headers=headers, data=json.dumps({"value": payload}))
    if r.status_code >= 400:
        raise RuntimeError(f"Ingestion failed: {r.status_code}, {r.text}")
    print(f"Ingested {len(payload)} records (heading + summary + detailed).")


# --------------------------
# Entry
# --------------------------


def main(config):
    delete_index_if_exists(config)
    create_hybrid_index(config)

    docs = load_docs_from_folder("src/docs")
    if not docs:
        raise RuntimeError(
            "No documents found in ./docs — please add at least one .txt file."
        )

    ingest_docs(config, docs)
