import numpy as np
import tiktoken
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from utils import load_docs_from_folder


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main(config):
    client = EmbeddingsClient(
        endpoint=f"{config['AZURE_OPENAI_RESOURCE_URL']}openai/deployments/embeddings",
        credential=AzureKeyCredential(config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_KEY"]),
    )

    # Tokenizer for counting
    enc = tiktoken.get_encoding("cl100k_base")
    max_chunk_size = 1000  # tokens
    threshold = 0.80  # similarity threshold

    docs = load_docs_from_folder("src/docs")

    for doc in docs:
        print(f"\n=== Processing doc: '{doc['title']}' ===")

        # Step 1: split by paragraphs
        paras = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]

        # Step 2: embed each paragraph
        embeddings = [
            client.embed(input=[p], model="text-embedding-3-small").data[0].embedding
            for p in paras
        ]

        # Step 3: semantic grouping + max size
        semantic_chunks = []
        current_chunk = [paras[0]]
        current_tokens = len(enc.encode(paras[0]))
        prev_emb = embeddings[0]

        for para, emb in zip(paras[1:], embeddings[1:]):
            para_tokens = len(enc.encode(para))
            sim = cosine_sim(prev_emb, emb)

            # check semantic similarity and token budget
            if sim >= threshold and (current_tokens + para_tokens) <= max_chunk_size:
                # merge into current chunk
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                # flush current chunk
                semantic_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

            prev_emb = emb

        if current_chunk:
            semantic_chunks.append("\n\n".join(current_chunk))

        # Step 4: print result
        for i, chunk in enumerate(semantic_chunks, 1):
            tokens = len(enc.encode(chunk))
            print(f"\n[Semantic Chunk {i}] {chunk[:300]}...")
            print(f"Token length: {tokens}")
