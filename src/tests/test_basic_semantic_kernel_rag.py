from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelFunction, KernelArguments
import asyncio

from models.azure import AzureOpenAIModel
from models.watson import WatsonXModel
from utils.parsers import dump_json


async def run_agentic_rag(config):
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="azure-gpt",
            deployment_name=config["AZURE_OPENAI_CHAT_DEPLOYMENT_MODEL"],
            endpoint=config["AZURE_OPENAI_CHAT_DEPLOYMENT_URL"],
            api_key=config["AZURE_OPENAI_CHAT_DEPLOYMENT_KEY"],
            api_version=config["AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION"],
        )
    )

    # native steps (no registration in v1.x)
    def guardrails_check(q: str):
        r = WatsonXModel.guardrails_check(
            [{"role": "user", "content": [{"type": "text", "text": q}]}]
        )
        return r, ("unsafe" in r.lower())

    def search_docs(q: str):
        return AzureOpenAIModel.azure_search({"query": q, "use_vectors": True})

    # prompt function (v1.x)
    rag_prompt = """
        You are a helpful banking assistant.
        Use ONLY this context to answer:

        {{$search_results}}

        Question: {{$input}}

        If the context is insufficient, say: "I don't have enough information to answer that."
    """

    rag_fn: KernelFunction = KernelFunction.from_prompt(
        function_name="generate_answer",
        plugin_name="rag",
        prompt=rag_prompt,
    )

    query = "What are the trading hours for the TSX?"

    _, is_unsafe = guardrails_check(query)
    if is_unsafe:
        print(f"❌ FAIL: Guardrails incorrectly blocked query: {query}")

        return

    docs = search_docs(query)
    if not docs:
        print(f"❌ FAIL: No results retrieved for query: {query}")

        return

    context_text = "\n".join(f"- {d['content']}" for d in docs)

    args = KernelArguments(input=query, search_results=context_text)
    ans = await rag_fn.invoke(kernel, arguments=args)  # v1.x invoke

    print(f"✅ PASS with response: {dump_json(getattr(ans, 'value', ans))}")


def main(config):
    asyncio.run(run_agentic_rag(config))
