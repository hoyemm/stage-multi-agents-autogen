import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

endpoint = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key = os.getenv("api_key")

model_client = OpenAIChatCompletionClient(
    model=deployment_name,
    api_key=api_key,
    base_url=endpoint,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    },
)

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message=(
        "Tu es un assistant IA utile et concis. "
        "Si l'utilisateur dit 'stop' ou 'au revoir', termine ta réponse par TERMINATE."
    ),
)

# input() classique = ta vraie saisie clavier dans le terminal
async def real_input(prompt: str, cancellation_token=None) -> str:
    return input(f"\n{prompt} ")

user_proxy = UserProxyAgent(
    name="user_proxy",
    input_func=real_input,   # <-- ici, plus de triche : c'est vraiment TOI qui réponds
)


async def main():
    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination,
    )

    # On lance la conversation en démarrant par le user_proxy (donc par TOI)
    stream = team.run_stream(task="Bonjour, j'ai une question.")
    async for message in stream:
        if hasattr(message, "source") and hasattr(message, "content"):
            print(f"\n{message.source}: {message.content}")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())