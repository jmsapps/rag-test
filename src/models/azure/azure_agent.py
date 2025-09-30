import asyncio, time
from typing import Annotated, Any, Dict, List, Optional, Tuple
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory

from models.azure.azure_openai import AzureOpenAIModel
from models.watson import WatsonXModel
from utils.parsers import dump_json


class _RunLogger:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.tools: List[Dict[str, Any]] = []
        self.user: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}

    def add(self, name: str, args: Dict[str, Any], result: Any, tstart: float) -> None:
        self.tools.append(
            {
                "tool": name,
                "args": args,
                "result": result,
                "duration_ms": round((time.time() - tstart) * 1000, 1),
            }
        )

    def finalize(self, resp_obj: Any, text: str) -> Dict[str, Any]:
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


class _Tools:
    def __init__(self, logger: _RunLogger):
        self.logger = logger

    @kernel_function(description="Guardrails check: returns 'safe' or 'unsafe'")
    def guardrails_check(self, q: Annotated[str, "User query"]) -> str:
        t0 = time.time()
        r = WatsonXModel.guardrails_check(
            [{"role": "user", "content": [{"type": "text", "text": q}]}]
        )
        out = "unsafe" if "unsafe" in r.lower() else "safe"
        self.logger.add("guardrails_check", {"q": q}, out, t0)
        return out

    @kernel_function(
        description="Vector search over indexed docs; returns JSON list of docs"
    )
    def search_docs(self, q: Annotated[str, "User query"]) -> str:
        t0 = time.time()
        docs = AzureOpenAIModel.azure_search({"query": q, "use_vectors": True})
        preview = {"count": len(docs)}

        if docs:
            preview["items"] = []
            for doc in docs:
                preview["items"].append(
                    {
                        "title": doc.get("title"),
                        "chunk_type": doc.get("chunk_type", "content"),  # fallback
                        "content_preview": (doc.get("content", "")[:160] + "..."),
                    }
                )

        self.logger.add("search_docs", {"q": q}, preview, t0)
        return dump_json(docs)


class AzureAgentModel:
    """Modern agentic RAG using SK function-calling agent."""

    def __init__(
        self,
        config: Dict[str, str],
        *,
        instructions: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.config = config
        self.logger = _RunLogger()

        self.service = AzureChatCompletion(
            deployment_name=config["AZURE_OPENAI_CHAT_DEPLOYMENT_MODEL"],
            endpoint=config["AZURE_OPENAI_CHAT_DEPLOYMENT_URL"],
            api_key=config["AZURE_OPENAI_CHAT_DEPLOYMENT_KEY"],
            api_version=config["AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION"],
        )

        settings = OpenAIChatPromptExecutionSettings()
        settings.temperature = temperature

        if top_p is not None:
            settings.top_p = top_p

        if max_tokens is not None:
            settings.max_tokens = max_tokens

        system_message = instructions or (
            "You must ALWAYS call `guardrails_check` first. "
            "NEVER answer directly from your own knowledge. "
            "If 'unsafe', refuse and suggest allowed help. "
            "If 'safe', you MUST call `search_docs` to retrieve documents. "
            "Parse the JSON returned. "
            "Use `summary` chunks for broad context, `heading` chunks for "
            "navigation, and `detailed` chunks for exact answers ONLY if "
            "`heading` and `summary` are insufficient. "
            "If only `content` is present (older indexes), use it as before. "
            "Answer ONLY from the retrieved docs. "
            "If insufficient, say: 'I'm sorry, I couldn't find any information "
            "about that. Can you please reword your question?' "
            "If irrelevant, kindly let the user know that you can only answer "
            "questions related to HappyTrade."
        )

        self.agent = ChatCompletionAgent(
            service=self.service,
            name="Banking-RAG",
            instructions=system_message,
            plugins=[_Tools(self.logger)],
            arguments=KernelArguments(settings),
        )

        self.chat_history = ChatHistory(system_message=system_message)

    def reset_chat_history(self, system_message: Optional[str] = None) -> None:
        if system_message is None:
            system_message = self.agent.instructions
        self.chat_history = ChatHistory(system_message=system_message)

    async def async_ask(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Async ask. Returns (answer_text, full_trace_dict)."""
        self.chat_history.add_user_message(question)
        self.logger.t0 = time.time()
        self.logger.user = {"question": question}
        self.logger.tools.clear()

        try:
            resp = await asyncio.wait_for(
                self.agent.get_response(self.chat_history), timeout=15
            )
        except asyncio.TimeoutError:
            return "", {"error": "timeout", "tools": self.logger.tools}

        # extract plain text
        text = getattr(resp, "value", None) or " ".join(
            getattr(i, "text", "")
            for i in getattr(resp, "items", [])
            if getattr(i, "text", None)
        )

        if text:
            self.chat_history.add_assistant_message(text)

        trace = self.logger.finalize(resp, text or "")
        return text or "", trace

    def ask(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Sync wrapper for tests."""
        return asyncio.run(self.async_ask(question))
