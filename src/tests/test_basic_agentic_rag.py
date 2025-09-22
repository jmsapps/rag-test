import asyncio, json, time, sys
from typing import Annotated
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.agents import ChatCompletionAgent

from models.azure import AzureOpenAIModel
from models.watson import WatsonXModel
from utils.parsers import dump_json


class RunLogger:
    def __init__(self):
        self.t0 = time.time()
        self.tools = []
        self.user = {}
        self.meta = {}

    def add(self, name, args, result, tstart):
        self.tools.append(
            {
                "tool": name,
                "args": args,
                "result": result,
                "duration_ms": round((time.time() - tstart) * 1000, 1),
            }
        )

    def finalize(self, resp_obj, text):
        def get(o, p, default=None):
            cur = o
            for k in p.split("."):
                cur = (
                    (cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None))
                    if cur is not None
                    else None
                )
            return default if cur is None else cur

        return {
            "started_at": self.t0,
            "elapsed_ms": round((time.time() - self.t0) * 1000, 1),
            "user": self.user,
            "tools": self.tools,
            "answer": {
                "text": text,
                "model": get(resp_obj, "ai_model_id")
                or get(resp_obj, "inner_content.model"),
                "id": get(resp_obj, "metadata.id") or get(resp_obj, "inner_content.id"),
                "created": get(resp_obj, "metadata.created")
                or get(resp_obj, "inner_content.created"),
                "finish_reason": str(
                    get(resp_obj, "finish_reason")
                    or get(resp_obj, "inner_content.choices.0.finish_reason")
                    or ""
                ),
                "usage": {
                    "prompt_tokens": get(resp_obj, "metadata.usage.prompt_tokens")
                    or get(resp_obj, "inner_content.usage.prompt_tokens"),
                    "completion_tokens": get(
                        resp_obj, "metadata.usage.completion_tokens"
                    )
                    or get(resp_obj, "inner_content.usage.completion_tokens"),
                    "total_tokens": get(resp_obj, "metadata.usage.total_tokens")
                    or get(resp_obj, "inner_content.usage.total_tokens"),
                },
            },
        }


class Tools:
    def __init__(self, logger):
        self.logger = logger

    @kernel_function(description="Guardrails check: returns 'safe' or 'unsafe'")
    def guardrails_check(self, q: Annotated[str, "User query"]) -> str:
        t = time.time()
        r = WatsonXModel.guardrails_check(
            [{"role": "user", "content": [{"type": "text", "text": q}]}]
        )
        out = "unsafe" if "unsafe" in r.lower() else "safe"
        self.logger.add("guardrails_check", {"q": q}, out, t)
        return out

    @kernel_function(
        description="Vector search over indexed docs; returns JSON list of docs"
    )
    def search_docs(self, q: Annotated[str, "User query"]) -> str:
        t = time.time()
        docs = AzureOpenAIModel.azure_search({"query": q, "use_vectors": True})
        # log a compact preview
        preview = {"count": len(docs)}
        if docs:
            preview["first"] = {
                "title": docs[0].get("title"),
                "content_preview": (docs[0].get("content", "")[:160] + "..."),
            }
        self.logger.add("search_docs", {"q": q}, preview, t)
        return json.dumps(docs, ensure_ascii=False)


async def run_agentic_rag(config):
    logger = RunLogger()
    svc = AzureChatCompletion(
        deployment_name=config["AZURE_OPENAI_CHAT_DEPLOYMENT_MODEL"],
        endpoint=config["AZURE_OPENAI_CHAT_DEPLOYMENT_URL"],
        api_key=config["AZURE_OPENAI_CHAT_DEPLOYMENT_KEY"],
        api_version=config["AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION"],
    )
    settings = OpenAIChatPromptExecutionSettings()
    # settings.temperature = 0.2; settings.max_tokens = 256

    agent = ChatCompletionAgent(
        service=svc,
        name="Banking-RAG",
        instructions=(
            "Always call `guardrails_check` first. "
            "If 'unsafe', refuse and suggest allowed help. "
            "If 'safe', call `search_docs`, parse JSON, join 'content' fields, "
            "answer ONLY from that context; if insufficient, say so."
        ),
        plugins=[Tools(logger)],
        arguments=KernelArguments(settings),
    )

    # === test input ===
    question = "What are the trading hours for the TSX?"
    logger.user = {"question": question}

    resp = await agent.get_response(question)
    text = getattr(resp, "value", None) or " ".join(
        getattr(i, "text", "")
        for i in getattr(resp, "items", [])
        if getattr(i, "text", None)
    )
    run = logger.finalize(resp, text or "")

    # === assertions ===
    failures = []
    # 1) guardrails must be called
    gr = next((t for t in run["tools"] if t["tool"] == "guardrails_check"), None)
    if not gr:
        failures.append("guardrails_check not called")
    # 2) if safe → search_docs must be called
    if gr and gr["result"] == "safe":
        if not any(t["tool"] == "search_docs" for t in run["tools"]):
            failures.append("search_docs not called after safe guardrails")
    # 3) answer must be non-empty
    if not run["answer"]["text"]:
        failures.append("empty answer")
    # 4) usage sanity (if available)
    usage = run["answer"]["usage"]
    if usage["total_tokens"] and usage["total_tokens"] > 8192:
        failures.append(f"token budget exceeded: {usage['total_tokens']}")
    # 5) latency budget
    if run["elapsed_ms"] > 15000:
        failures.append(f"slow run: {run['elapsed_ms']} ms")

    # print summary
    print(dump_json(run))
    if failures:
        print("\n❌ FAIL:", "; ".join(failures))
        sys.exit(1)
    else:
        print("\n✅ PASS")
        sys.exit(0)


def main(config):
    asyncio.run(run_agentic_rag(config))
