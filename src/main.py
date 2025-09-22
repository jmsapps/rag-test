from models.azure import AzureAgentModel
from globals import config


def chat(azure_agent: AzureAgentModel) -> bool:
    try:
        user_input = input("User: ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    response, _ = azure_agent.ask(user_input)

    if response:
        print(f"Azure Agent: {response}")

    return True


def main() -> None:
    azure_agent = AzureAgentModel(config, max_tokens=8192)
    chatting = True
    while chatting:
        chatting = chat(azure_agent)


if __name__ == "__main__":
    main()
