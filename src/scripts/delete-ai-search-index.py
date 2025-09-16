import requests
from pprint import pprint


def main(config):
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}?api-version=2023-11-01"

    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
    }

    try:
        response = requests.delete(url, headers=headers)

        if response.status_code == 204:
            print(f"Index '{config['AZURE_SEARCH_API_INDEX']}' deleted successfully.")
        elif response.status_code == 404:
            print(f"Index '{config['AZURE_SEARCH_API_INDEX']}' not found.")
        else:
            print(f"Unexpected status: {response.status_code}")
            pprint(vars(response))

    except Exception as e:
        print(f"Error: {e}")
