import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Config(TypedDict):
    WATSON_X_API_URL: str
    WATSON_X_API_KEY: str
    WATSON_X_PROJECT_ID: str


config: Config = {
    "WATSON_X_API_URL": os.getenv("WATSON_X_API_URL"),
    "WATSON_X_API_KEY": os.getenv("WATSON_X_API_KEY"),
    "WATSON_X_PROJECT_ID": os.getenv("WATSON_X_PROJECT_ID"),
}
