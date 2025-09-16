import json
import requests
from globals import config
from models import WatsonX


def guardrails_check(query: str) -> str:
    """Run WatsonX guardrails to classify a text query."""
    if "watsonx" in config.get("BYPASS", []):

        return "safe"

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": query}],
        }
    ]

    response = WatsonX.get_inference_model().chat(messages=messages)

    return response["choices"][0]["message"]["content"].strip() or ""


def azure_search(query: str):
    """Run a search query against Azure Cognitive Search."""
    url = f"{config['AZURE_SEARCH_API_URL']}/indexes/{config['AZURE_SEARCH_API_INDEX']}/docs/search?api-version=2023-11-01"

    headers = {
        "Content-Type": "application/json",
        "api-key": config["AZURE_SEARCH_API_PRIMARY_ADMIN_KEY"],
    }

    payload = {"search": query, "top": 3}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    response.raise_for_status()

    return response.json().get("value", [])


def main(**_):
    # Simple deterministic test cases
    test_cases = {
        "minimum deposit": "The minimum deposit for a savings account is $100.",
        "transfer fees": "Transfers between InvestorLine accounts are free.",
        "trading hours": "Trades are available 9:30 AM – 4:00 PM EST.",
    }

    all_passed = True

    for query, expected_substring in test_cases.items():
        print(f"\nTesting query: {repr(query)} ...")

        # Step 1: Guardrails
        guardrails_result = guardrails_check(query)

        print(f"Guardrails result: {repr(guardrails_result)}")

        if "unsafe" in guardrails_result.lower():
            print(f"FAIL: Guardrails blocked an unsafe query: {repr(query)}")
            all_passed = False

            continue

        # Step 2: Search
        try:
            docs = azure_search(query)
        except Exception as e:
            print(f"Error: {e}")
            all_passed = False

            continue

        if not docs:
            print(f"FAIL: No results for {repr(query)}")
            all_passed = False

            continue

        # Step 3: Check if expected content is in results
        combined_content = " ".join(doc["content"] for doc in docs)

        if expected_substring.lower() in combined_content.lower():
            print(
                f"PASS: Found expected content {repr(expected_substring.lower())} for {repr(query)}"
            )
        else:
            print(f"FAIL: Did not find expected content for {repr(query)}")
            print(f"Got: {combined_content}")
            all_passed = False

    if all_passed:
        print("\nAll basic RAG tests passed!")
    else:
        print("\nSome tests failed. Check output above.")
