from typing import TypedDict


class WatsonInferenceModelMessageContent(TypedDict):
    type: str
    text: str


class WatsonInferenceModelMessage(TypedDict):
    role: str
    content: list[WatsonInferenceModelMessageContent]
