from models.azure import AzureAgentModel
from globals import config


def chat(azure_agent: AzureAgentModel) -> bool:
    try:
        user_input = input("\nUser: ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    response, trace = azure_agent.ask(user_input)

    if response:
        print(f"\nAzure Agent: {response}")

    if trace.get("tools"):
        print("\n--- Agent Internal Trace ---")
        for t in trace["tools"]:
            print(f"[{t['tool']}] q={t['args'].get('q')} → {t['result']}")
        print("-----------------------------\n")
    else:
        print(f"No tools found: {trace}")

    return True


def main() -> None:
    azure_agent = AzureAgentModel(config, max_tokens=8192)
    chatting = True
    while chatting:
        chatting = chat(azure_agent)


if __name__ == "__main__":
    main()
