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

Sécurités mises en place (gestion du contexte / coûts) :
  - Historique de chaque agent limité aux N derniers messages
    (BufferedChatCompletionContext) pour éviter de dépasser la fenêtre de
    contexte du modèle sur les tâches longues (Action corrective : tronquage
    des messages les plus anciens avant l'appel API).
  - `max_tokens` strict configuré sur le client de modèle : limite la taille
    de CHAQUE réponse générée par un agent (contrôle des coûts).
  - `TokenUsageTermination` : coupe la conversation si le total de tokens
    consommés (prompt + completion, cumulés sur toute l'équipe) dépasse un
    budget fixé (contrôle des coûts au niveau de la conversation entière).
  - `RepeatedContentTermination` (maison) : détecte qu'un agent répète un
    message quasi identique à son message précédent (boucle) et arrête la
    conversation immédiatement (early stopping anti-boucle).
  - Terminaison aussi sur mot-clé "TERMINATE" OU nombre max de messages
    atteint (anti-boucle infinie / anti-surconsommation de tokens).
  - Exécution du code exclusivement dans Docker (jamais sur l'hôte).
  - Journalisation (logging) de tous les échanges + de l'usage de tokens
    dans un fichier .jsonl, pour analyse et suivi des coûts.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Sequence

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
    TokenUsageTermination,
)
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.model_context import BufferedChatCompletionContext

load_dotenv()

endpoint = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key = os.getenv("api_key")

# --- Paramètres anti-boucle / anti-dépassement de contexte -----------------

# Nombre max de messages échangés dans l'équipe avant coupure forcée
# (protection anti-boucle infinie / anti-surconsommation de tokens).
MAX_MESSAGES = 12

# Nombre de messages passés que CHAQUE agent conserve dans son propre
# contexte envoyé au modèle. Au-delà, les plus anciens sont tronqués
# automatiquement. Alternative possible : utiliser un modèle avec une
# fenêtre de contexte plus grande (ex. 128k), mais cela ne résout pas le
# coût/latence croissants ; le tronquage reste utile dans tous les cas.
CONTEXT_BUFFER_SIZE = 10

# --- Paramètres de contrôle des coûts (nouveau) -----------------------------

# Limite stricte du nombre de tokens générés PAR appel API (par réponse
# d'agent). Empêche un agent de produire une réponse démesurée.
MAX_TOKENS_PER_CALL = 800

# Budget total de tokens (prompt + completion, cumulés sur toute l'équipe)
# avant coupure automatique de la conversation. Sert de garde-fou de coût
# en plus de MAX_MESSAGES (une conversation courte mais très verbeuse
# dépasserait quand même ce budget).
MAX_TOTAL_TOKENS = 20000

# Nombre de répétitions quasi identiques consécutives (même agent) tolérées
# avant arrêt anticipé (early stopping anti-boucle).
REPEAT_THRESHOLD = 2

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
        max_tokens=MAX_TOKENS_PER_CALL,  # limite de coût par appel
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
            "Agent PLANIFICATEUR. Décompose la demande utilisateur en un plan "
            "numéroté, concis (3-6 étapes max), réalisable par un Codeur puis "
            "un Réviseur. Pas de code, pas de justification longue. "
            "Réponds uniquement par la liste numérotée. N'écris jamais TERMINATE."
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
            "Agent CODEUR. Écris uniquement le code Python demandé dans un seul "
            "bloc ```python ... ```, sans input() (le code s'exécute sans "
            "interaction), avec un print() du résultat. Pas de longue "
            "explication autour du code. Si le réviseur signale une erreur, "
            "renvoie une version corrigée complète, uniquement le code corrigé. "
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
            "Agent RÉVISEUR. Tu reçois le code du codeur ET son résultat "
            "d'exécution réel (Docker). Vérifie : erreurs d'exécution, failles "
            "de sécurité, style. Sois concis.\n"
            "- Si correction nécessaire : liste les problèmes en quelques "
            "phrases courtes, sans réécrire le code toi-même.\n"
            "- Si le code est correct et sûr : réponds uniquement "
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


# ---------------------------------------------------------------------------
# Early stopping anti-boucle : détecte qu'un agent répète un message
# quasi identique à son message précédent (ex. codeur/réviseur qui tournent
# en rond sans converger) et force l'arrêt de la conversation.
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


class RepeatedContentTermination(TerminationCondition):
    """Arrête la conversation si un même agent envoie 2 messages quasi
    identiques (normalisés) d'affilée dans sa propre série de messages.
    Sert de garde-fou contre les boucles codeur<->réviseur qui ne
    convergent pas, en plus de MaxMessageTermination et TokenUsageTermination.
    """

    def __init__(self, repeat_threshold: int = REPEAT_THRESHOLD):
        self._terminated = False
        self._repeat_threshold = repeat_threshold
        self._last_by_source: dict[str, tuple[str, int]] = {}

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")

        for message in messages:
            content = getattr(message, "content", None)
            source = getattr(message, "source", None)
            if not isinstance(content, str) or not source:
                continue

            normalized = _normalize(content)
            prev_text, prev_count = self._last_by_source.get(source, ("", 0))

            if normalized and normalized == prev_text:
                count = prev_count + 1
            else:
                count = 1
            self._last_by_source[source] = (normalized, count)

            if count >= self._repeat_threshold:
                self._terminated = True
                return StopMessage(
                    content=(
                        f"Arrêt anticipé : l'agent '{source}' a répété un "
                        f"message quasi identique {count} fois de suite "
                        "(boucle détectée)."
                    ),
                    source="RepeatedContentTermination",
                )
        return None

    async def reset(self) -> None:
        self._terminated = False
        self._last_by_source = {}


def build_termination_condition() -> TerminationCondition:
    """Combine toutes les conditions d'arrêt : succès (TERMINATE), sécurité
    (nb max de messages), coût (budget de tokens) et anti-boucle."""
    return (
        TextMentionTermination("TERMINATE")
        | MaxMessageTermination(MAX_MESSAGES)
        | TokenUsageTermination(max_total_token=MAX_TOTAL_TOKENS)
        | RepeatedContentTermination(REPEAT_THRESHOLD)
    )


def make_logger(log_file: str):
    """
    Retourne une fonction qui ajoute chaque message à un fichier .jsonl
    (une ligne JSON par message), pour permettre une analyse ultérieure
    des échanges (Critère : logs enregistrés dans un fichier), y compris
    l'usage de tokens quand il est disponible (suivi des coûts).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_message(source: str, content: str, usage: dict | None = None):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "content": content,
        }
        if usage:
            entry["usage"] = usage
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return log_message


async def build_team(model_client: OpenAIChatCompletionClient) -> tuple[
    SelectorGroupChat, DockerCommandLineCodeExecutor
]:
    """Construit l'équipe complète (agents + SelectorGroupChat) avec toutes
    les protections (contexte, coût, anti-boucle). Utilisé à la fois par le
    script CLI (main) et par l'interface Streamlit, pour éviter toute
    duplication / divergence de configuration."""
    planificateur = build_planner(model_client)
    codeur = build_coder(model_client)
    reviseur = build_reviewer(model_client)
    executeur, docker_executor = await build_docker_executor_agent()

    termination = build_termination_condition()

    team = SelectorGroupChat(
        [planificateur, codeur, executeur, reviseur],
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )
    return team, docker_executor


def extract_usage(message) -> dict | None:
    """Extrait le nombre de tokens (prompt/completion) d'un message d'agent,
    quand l'info est disponible (attribut models_usage), pour le logger et
    suivre les coûts réels de la conversation."""
    usage = getattr(message, "models_usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


async def main():
    model_client = build_model_client()
    team, docker_executor = await build_team(model_client)
    log_message = make_logger(LOG_FILE)

    task = input("\nDécrivez votre demande : ")
    log_message("user", task)

    try:
        # On consomme le flux nous-mêmes (au lieu d'utiliser seulement Console)
        # afin de pouvoir logger chaque message (+ usage tokens) dans le
        # fichier en plus de l'affichage terminal.
        async for message in team.run_stream(task=task):
            if hasattr(message, "source") and hasattr(message, "content"):
                print(f"\n---------- {message.source} ----------\n{message.content}")
                log_message(message.source, str(message.content), extract_usage(message))
            elif isinstance(message, StopMessage):
                print(f"\n---------- ARRÊT ----------\n{message.content}")
                log_message(message.source, str(message.content))
    finally:
        await docker_executor.stop()
        await model_client.close()
        print(f"\nConversation enregistrée dans : {LOG_FILE}")


if __name__ == "__main__":
    asyncio.run(main())