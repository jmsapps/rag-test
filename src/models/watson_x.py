from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from globals import config, Config


class WatsonX:
    @staticmethod
    def get_credentials(config: Config):
        return Credentials(
            url=config["WATSON_X_API_URL"], api_key=config["WATSON_X_API_KEY"]
        )

    @staticmethod
    def get_inference_model():
        return ModelInference(
            model_id="meta-llama/llama-guard-3-11b-vision",
            credentials=WatsonX.get_credentials(config),
            project_id=config["WATSON_X_PROJECT_ID"],
        )
