"""
Équipe multi-agents AutoGen (autogen_agentchat) :
  1. Planificateur : décompose la demande en sous-tâches
  2. Codeur        : écrit le code Python demandé
  3. Réviseur      : relit le code, propose des corrections, valide
  4. Exécuteur     : exécute le code validé dans un conteneur Docker isolé

Sécurités mises en place :
  - Terminaison sur mot-clé "TERMINATE" OU nombre max de messages atteint
    (équivalent moderne de max_consecutive_auto_reply / is_termination_msg)
  - Exécution du code exclusivement dans Docker (équivalent de use_docker=True)
"""

import os
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

load_dotenv()

endpoint = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key = os.getenv("api_key")

# Nombre max de messages échangés dans l'équipe avant coupure forcée
# (protection anti-boucle infinie / anti-surconsommation de tokens)
MAX_MESSAGES = 12


def build_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
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


# ---------------------------------------------------------------------------
# Tâche 1 : Agent Planificateur
# ---------------------------------------------------------------------------
def build_planner(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="planificateur",
        model_client=model_client,
        system_message=(
            "Tu es un agent PLANIFICATEUR. Ton rôle est de décomposer la demande "
            "de l'utilisateur en un plan d'action clair, numéroté, en plusieurs "
            "étapes concrètes et réalisables par les agents suivants (un Codeur "
            "et un Réviseur). Ne rédige pas de code toi-même. "
            "Termine toujours ta réponse par la liste d'étapes numérotées "
            "(1., 2., 3., ...). N'écris jamais TERMINATE."
        ),
    )


# ---------------------------------------------------------------------------
# Tâche 2 : Agent Codeur
# ---------------------------------------------------------------------------
def build_coder(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="codeur",
        model_client=model_client,
        system_message=(
            "Tu es un agent CODEUR. En te basant sur le plan fourni par le "
            "planificateur, écris du code Python dans un bloc ```python ... ```. "
            "Si le réviseur signale des erreurs ou des améliorations, corrige "
            "ton code en conséquence et renvoie une version mise à jour complète. "
            "N'écris jamais TERMINATE."
        ),
    )


# ---------------------------------------------------------------------------
# Tâche 2 (suite) : Agent Réviseur
# ---------------------------------------------------------------------------
def build_reviewer(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="reviseur",
        model_client=model_client,
        system_message=(
            "Tu es un agent RÉVISEUR de code. Analyse le code produit par le "
            "codeur : repère les erreurs, les failles de sécurité (ex. suppression "
            "de fichiers, accès réseau non contrôlé, commandes système dangereuses) "
            "et les problèmes de style. "
            "- Si le code doit être corrigé, explique précisément ce qui ne va pas.\n"
            "- Si le code est correct et sûr, réponds uniquement par : "
            "'Code validé. TERMINATE'"
        ),
    )


# ---------------------------------------------------------------------------
# Tâche 4 : Exécuteur de code isolé (Docker)
# ---------------------------------------------------------------------------
async def build_docker_executor_agent() -> tuple[CodeExecutorAgent, DockerCommandLineCodeExecutor]:
    """
    Crée un agent qui exécute le code validé dans un conteneur Docker isolé.
    Nécessite : pip install "autogen-ext[docker]"  +  Docker démarré sur la machine.
    Le code n'est JAMAIS exécuté sur l'hôte : le conteneur est jetable et
    n'a pas accès au reste du système de fichiers.
    """
    docker_executor = DockerCommandLineCodeExecutor(
        image="python:3-slim",
        work_dir="coding_sandbox",   # dossier local monté dans le conteneur
        timeout=60,
    )
    await docker_executor.start()

    executor_agent = CodeExecutorAgent(
        name="executeur_sandbox",
        code_executor=docker_executor,
    )
    return executor_agent, docker_executor


async def main():
    model_client = build_model_client()

    planificateur = build_planner(model_client)
    codeur = build_coder(model_client)
    reviseur = build_reviewer(model_client)
    executeur, docker_executor = await build_docker_executor_agent()

    # Tâche 3 : condition de sortie combinée
    #  - mot-clé TERMINATE émis par le réviseur
    #  - OU nombre max de messages atteint (anti-boucle infinie / anti-quota)
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(MAX_MESSAGES)

    team = RoundRobinGroupChat(
        [planificateur, codeur, reviseur, executeur],
        termination_condition=termination,
    )

    task = input("\nDécrivez votre demande : ")

    try:
        # Console() affiche joliment le flux de messages au fur et à mesure
        await Console(team.run_stream(task=task))
    finally:
        await docker_executor.stop()
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())