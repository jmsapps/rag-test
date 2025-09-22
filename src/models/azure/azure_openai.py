from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from semantic_kernel.contents import ChatHistory
from openai import AzureOpenAI
import json
import requests

from globals import config, Config
from .types import (
    AzureSearchParams,
    AzureOpenAIGenerateParams,
    AzureOpenAIGeneratePromptParams,
)


class AzureOpenAIModel:
    @staticmethod
    def azure_search(params: AzureSearchParams):
        """Run a search query against Azure Cognitive Search."""
        url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/search?api-version=2023-11-01"

        headers = {
            "Content-Type": "application/json",
            "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
        }

        payload = {"search": params["query"], "top": params.get("top", 3)}

        if params.get("use_vectors") and params["use_vectors"] is True:
            embedding = AzureOpenAIModel.azure_openai_generate_embedding(
                params["query"]
            )

            payload = {
                "count": True,
                "select": "title, content",
                "vectorQueries": [
                    {
                        "kind": "vector",
                        "vector": embedding,
                        "fields": "embedding",
                        "k": params.get("top", 3),
                    },
                ],
            }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code >= 400:
            raise RuntimeError(
                f"Status code: {response.status_code}. Error: {response.text}"
            )

        return response.json().get("value", [])

    @staticmethod
    def azure_openai_generate_prompt(params: AzureOpenAIGeneratePromptParams):
        context = "\n".join(f"- {doc['content']}" for doc in params["context_docs"])

        instructions = params.get(
            "instructions",
            "Answer the following query using only the context below.",
        )

        return (
            f"{instructions}\n\n"
            f"Context:\n{context}\n\n"
            f"Query: \n{params["query"]}"
        )

    @staticmethod
    def azure_openai_generate(params: AzureOpenAIGenerateParams):
        """Call Azure OpenAI to synthesize an answer from retrieved documents."""
        client: AzureOpenAI = AzureOpenAI(
            api_key=config["AZURE_OPENAI_CHAT_DEPLOYMENT_KEY"],
            api_version=config["AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_CHAT_DEPLOYMENT_URL"],
        )

        max_tokens = (
            params["max_tokens"]
            if params.get("max_tokens")
            and params["max_tokens"] >= 0
            and params["max_tokens"] <= 10_000
            else 256
        )
        temperature = (
            params["temperature"]
            if params.get("temperature")
            and params["temperature"] >= 0.0
            and params["temperature"] <= 2.0
            else 1.0
        )
        top_p = (
            params["top_p"]
            if params.get("top_p") and params["top_p"] >= 0.0 and params["top_p"] <= 1.0
            else 1.0
        )

        response = client.chat.completions.create(
            model=config["AZURE_OPENAI_CHAT_DEPLOYMENT_MODEL"],
            messages=params["messages"],
            max_tokens=max_tokens,
            temperature=temperature,  # range 0.0 -> 2.0 = deterministic -> creative
            top_p=top_p,  # range 0.0 -> 1.0 = narrow filtering of next-word choices -> wide/no filtering
        )

        return response.choices[0].message.content.strip()

    @staticmethod
    def azure_openai_generate_embedding(text: str):
        """Generate an embedding vector for a single text chunk."""

        client = EmbeddingsClient(
            endpoint=f"{config["AZURE_OPENAI_RESOURCE_URL"]}openai/deployments/embeddings",
            credential=AzureKeyCredential(
                config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_KEY"]
            ),
        )

        response = client.embed(
            input=[text],
            model=config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_MODEL"],
        )

        return response.data[0].embedding
