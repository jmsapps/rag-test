import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Config(TypedDict):
    # GLOBAL
    BYPASS: list[str]
    ENVIRONMENT: str
    # IBM
    WATSONX_API_URL: str
    WATSONX_API_KEY: str
    WATSONX_PROJECT_ID: str
    # Azure
    AZURE_SEARCH_API_URL: str
    AZURE_SEARCH_API_PRIMARY_ADMIN_KEY: str
    AZURE_SEARCH_API_INDEX: str


config: Config = {
    "WATSONX_API_URL": os.getenv("WATSONX_API_URL"),
    "WATSONX_API_KEY": os.getenv("WATSONX_API_KEY"),
    "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID"),
    "ENVIRONMENT": os.getenv("ENVIRONMENT"),
    "BYPASS": os.getenv("BYPASS").split(","),
    "AZURE_SEARCH_API_URL": os.getenv("AZURE_SEARCH_API_URL"),
    "AZURE_SEARCH_API_PRIMARY_ADMIN_KEY": os.getenv(
        "AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"
    ),
    "AZURE_SEARCH_API_INDEX": os.getenv("AZURE_SEARCH_API_INDEX"),
}
