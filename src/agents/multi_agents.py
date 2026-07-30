"""
Équipe multi-agents AutoGen (autogen_agentchat) :
  1. Planificateur : décompose la demande en sous-tâches
  2. Codeur        : écrit le code Python demandé
  3. Exécuteur     : exécute le code dans un conteneur Docker isolé
  4. Réviseur      : relit le code ET son résultat d'exécution réel,
                     propose des corrections ou valide (TERMINATE)

NOTE SUR LE VOCABULAIRE (ancienne API pyautogen vs API actuelle autogen_agentchat) :
  - "GroupChat" + "GroupChatManager" (ancienne API) ==> ici "SelectorGroupChat"
    (API actuelle). SelectorGroupChat EST le GroupChat + son manager réunis :
    à chaque tour, un LLM "sélecteur" lit l'historique et choisit, parmi tous
    les participants, celui qui doit parler ensuite, selon les règles qu'on
    lui donne dans `selector_prompt` (= la "speaker selection method").
  - "max_consecutive_auto_reply" / "is_termination_msg" (ancienne API) ==>
    TextMentionTermination + MaxMessageTermination (API actuelle).
  - "use_docker=True" (ancienne API) ==> DockerCommandLineCodeExecutor
    (API actuelle).

Sécurités mises en place :
  - Terminaison sur mot-clé "TERMINATE" OU nombre max de messages atteint
    (anti-boucle infinie / anti-surconsommation de tokens)
  - Exécution du code exclusivement dans Docker (jamais sur l'hôte)
  - Historique de chaque agent limité aux N derniers messages
    (BufferedChatCompletionContext) pour éviter de dépasser la fenêtre de
    contexte du modèle sur les tâches longues
  - Journalisation (logging) de tous les échanges dans un fichier, pour analyse
"""

import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.model_context import BufferedChatCompletionContext

load_dotenv()

endpoint = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key = os.getenv("api_key")

# Nombre max de messages échangés dans l'équipe avant coupure forcée
# (protection anti-boucle infinie / anti-surconsommation de tokens)
MAX_MESSAGES = 12

# Nombre de messages passés que CHAQUE agent conserve dans son propre
# contexte envoyé au modèle. Au-delà, les plus anciens sont tronqués
# automatiquement. Alternative possible : utiliser un modèle avec une
# fenêtre de contexte plus grande (ex. 128k), mais cela ne résout pas le
# coût/latence croissants ; le tronquage reste utile dans tous les cas.
CONTEXT_BUFFER_SIZE = 10

# Dossier + fichier de logs des conversations (un fichier par exécution)
LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR, f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)


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
        description="Décompose la demande utilisateur en un plan d'action numéroté.",
        model_context=BufferedChatCompletionContext(buffer_size=CONTEXT_BUFFER_SIZE),
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
        description="Écrit et corrige le code Python à partir du plan et des retours du réviseur.",
        model_context=BufferedChatCompletionContext(buffer_size=CONTEXT_BUFFER_SIZE),
        system_message=(
            "Tu es un agent CODEUR. En te basant sur le plan fourni par le "
            "planificateur (ou sur les corrections demandées par le réviseur), "
            "écris du code Python complet dans un unique bloc ```python ... ```. "
            "Le code sera exécuté automatiquement juste après ton message : "
            "assure-toi qu'il s'exécute sans entrée interactive (n'utilise pas "
            "input()) et qu'il affiche un résultat clair via print(). "
            "Si le réviseur signale des erreurs après exécution, corrige ton "
            "code en conséquence et renvoie une version mise à jour complète. "
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
        description="Analyse le code ET son résultat d'exécution réel ; valide ou renvoie au codeur.",
        model_context=BufferedChatCompletionContext(buffer_size=CONTEXT_BUFFER_SIZE),
        system_message=(
            "Tu es un agent RÉVISEUR de code. Tu interviens APRÈS que le code du "
            "codeur a été exécuté dans un conteneur Docker isolé : tu disposes donc "
            "à la fois du code source et de son résultat d'exécution réel (sortie "
            "standard, erreurs éventuelles). "
            "Analyse les deux : repère les erreurs d'exécution, les failles de "
            "sécurité (ex. suppression de fichiers, accès réseau non contrôlé, "
            "commandes système dangereuses) et les problèmes de style. "
            "- Si le code a échoué à l'exécution OU doit être corrigé, explique "
            "précisément ce qui ne va pas afin que le codeur puisse corriger.\n"
            "- Si le code s'est exécuté avec succès, est correct et sûr, réponds "
            "uniquement par : 'Code validé. TERMINATE'"
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
        description="Exécute le code Python le plus récent dans un conteneur Docker isolé.",
    )
    return executor_agent, docker_executor


# ---------------------------------------------------------------------------
# Règles de prise de parole (speaker selection method) du SelectorGroupChat.
# Un LLM lit ce prompt + l'historique à chaque tour et choisit qui parle.
# ---------------------------------------------------------------------------
SELECTOR_PROMPT = """Tu choisis quel agent doit parler ensuite dans une équipe
qui transforme une demande utilisateur en code Python testé et validé.

Participants disponibles et leur rôle :
{roles}

Règles de sélection strictes :
1. Au tout début (aucun message d'agent encore), choisis toujours "planificateur".
2. Une fois qu'un plan a été donné par "planificateur", choisis "codeur".
3. Juste après un message de "codeur" contenant du code, choisis "executeur_sandbox"
   pour exécuter ce code (jamais le "reviseur" directement après le "codeur").
4. Juste après un message de "executeur_sandbox", choisis "reviseur".
5. Si "reviseur" demande des corrections (pas de TERMINATE), choisis "codeur"
   pour qu'il corrige, puis reviens à la règle 3.
6. Si "reviseur" a répondu "Code validé. TERMINATE", ne choisis personne
   d'autre : la conversation doit s'arrêter.

Historique de la conversation :
{history}

Réponds uniquement avec le nom exact d'un participant parmi : {participants}.
"""


def make_logger(log_file: str):
    """
    Retourne une fonction qui ajoute chaque message à un fichier .jsonl
    (une ligne JSON par message), pour permettre une analyse ultérieure
    des échanges (Critère : logs enregistrés dans un fichier).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_message(source: str, content: str):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "content": content,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return log_message


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

    # SelectorGroupChat = "GroupChat" + "GroupChatManager" de l'ancienne API.
    # allow_repeated_speaker=True : nécessaire ici car "codeur" peut reprendre
    # la parole plusieurs fois de suite lors des itérations de correction.
    team = SelectorGroupChat(
        [planificateur, codeur, executeur, reviseur],
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )

    log_message = make_logger(LOG_FILE)

    task = input("\nDécrivez votre demande : ")
    log_message("user", task)

    try:
        # On consomme le flux nous-mêmes (au lieu d'utiliser seulement Console)
        # afin de pouvoir logger chaque message dans le fichier en plus de
        # l'affichage terminal.
        async for message in team.run_stream(task=task):
            if hasattr(message, "source") and hasattr(message, "content"):
                print(f"\n---------- {message.source} ----------\n{message.content}")
                log_message(message.source, str(message.content))
    finally:
        await docker_executor.stop()
        await model_client.close()
        print(f"\nConversation enregistrée dans : {LOG_FILE}")


if __name__ == "__main__":
    asyncio.run(main())