from typing import TypedDict, Optional


class AzureSearchParams(TypedDict):
    query: str
    top: Optional[int]
    use_vector: Optional[bool]


class AzureOpenAIGeneratePromptParams(TypedDict):
    instructions: Optional[str]
    query: str
    context_docs: list[str]


class AzureOpenAIGenerateMessageContent(TypedDict):
    type: str
    text: str


class AzureOpenAIGenerateMessage(TypedDict):
    role: str
    content: list[AzureOpenAIGenerateMessageContent]


class AzureOpenAIGenerateParams(TypedDict):
    messages: list[AzureOpenAIGenerateMessage]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
