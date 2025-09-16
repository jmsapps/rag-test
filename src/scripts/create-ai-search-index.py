import requests
import json
from pprint import pprint


def main(config):
    url = f"{config["AZURE_SEARCH_API_URL"]}/indexes/{config["AZURE_SEARCH_API_INDEX"]}?api-version=2023-11-01"

    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
    }

    index_schema = {
        "name": "bank-faq",
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "sortable": True,
            },
            {"name": "content", "type": "Edm.String", "searchable": True},
        ],
    }

    try:
        response = requests.put(url, headers=headers, data=json.dumps(index_schema))

        if response.status_code >= 400:
            pprint(vars(response))
        else:
            print("Index created or updated:", response.json())
    except Exception as e:
        print(f"Error: {e}")
