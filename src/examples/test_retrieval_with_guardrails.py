from models.watson import WatsonXModel
from models.azure import AzureOpenAIModel


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
        guardrails_result = WatsonXModel.guardrails_check(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ]
        )

        print(f"Guardrails result: {repr(guardrails_result)}")

        if "unsafe" in guardrails_result.lower():
            print(f"FAIL: Guardrails blocked an unsafe query: {repr(query)}")
            all_passed = False

            continue

        # Step 2: Search
        try:
            docs = AzureOpenAIModel.azure_search({"query": query})
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
