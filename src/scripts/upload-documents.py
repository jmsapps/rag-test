import json
import requests
from pprint import pprint


def main(config):
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/index?api-version=2023-11-01"

    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
    }

    docs = {
        "value": [
            {
                "@search.action": "upload",
                "id": "1",
                "title": "Minimum Deposit",
                "content": "The minimum deposit for a savings account is $100.",
            },
            {
                "@search.action": "upload",
                "id": "2",
                "title": "Transfer Fees",
                "content": "Transfers between InvestorLine accounts are free.",
            },
            {
                "@search.action": "upload",
                "id": "3",
                "title": "Trading Hours",
                "content": "Trades are available 9:30 AM – 4:00 PM EST.",
            },
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(docs))

        if response.status_code >= 400:
            print(f"Error {response.status_code}:")
            pprint(vars(response))
        else:
            print("Documents uploaded:", response.json())
    except Exception as e:
        print(f"Error: {e}")
