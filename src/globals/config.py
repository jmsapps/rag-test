import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Config(TypedDict):
    WATSONX_API_URL: str
    WATSONX_API_KEY: str
    WATSONX_PROJECT_ID: str
    ENVIRONMENT: str
    BYPASS: list[str]


config: Config = {
    "WATSONX_API_URL": os.getenv("WATSONX_API_URL"),
    "WATSONX_API_KEY": os.getenv("WATSONX_API_KEY"),
    "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID"),
    "ENVIRONMENT": os.getenv("ENVIRONMENT"),
    "BYPASS": os.getenv("BYPASS").split(","),
}
