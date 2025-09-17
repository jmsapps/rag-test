from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from globals import config, Config
from .types import WatsonInferenceModelMessage


class WatsonXModel:
    @staticmethod
    def get_credentials(config: Config):
        return Credentials(
            url=config["WATSONX_API_URL"], api_key=config["WATSONX_API_KEY"]
        )

    @staticmethod
    def get_inference_model():
        return ModelInference(
            model_id="meta-llama/llama-guard-3-11b-vision",
            credentials=WatsonXModel.get_credentials(config),
            project_id=config["WATSONX_PROJECT_ID"],
        )

    @staticmethod
    def guardrails_check(messages: list[WatsonInferenceModelMessage]) -> str:
        """Run WatsonX guardrails to classify a basic text query."""

        if "watsonx" in config.get("BYPASS", []):
            return "safe"

        response = WatsonXModel.get_inference_model().chat(messages=messages)

        return response["choices"][0]["message"]["content"].strip() or ""
