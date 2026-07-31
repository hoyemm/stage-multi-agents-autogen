"""
Tests pour la gestion du contexte / des coûts / de l'anti-boucle.
Lancement : pytest -v test_context_management.py

Ces tests n'appellent PAS l'API Azure OpenAI : ils testent la logique pure
(RepeatedContentTermination, config des agents) avec des objets simulés,
pour rester rapides et gratuits à exécuter.
"""

import asyncio
import pytest

from autogen_agentchat.base import TerminatedException

from multi_agents import (
    RepeatedContentTermination,
    REPEAT_THRESHOLD,
    MAX_TOKENS_PER_CALL,
    CONTEXT_BUFFER_SIZE,
    build_planner,
    build_coder,
    build_reviewer,
)


class FakeMessage:
    """Simule un message d'agent (TextMessage) sans dépendre de l'API réelle."""

    def __init__(self, source: str, content: str):
        self.source = source
        self.content = content


def test_repeated_content_triggers_stop():
    """Deux messages quasi identiques d'affilée du même agent -> arrêt."""
    cond = RepeatedContentTermination(repeat_threshold=2)

    async def run():
        first = await cond([FakeMessage("codeur", "def add(a, b): return a + b")])
        assert first is None
        assert cond.terminated is False

        second = await cond([FakeMessage("codeur", "def add(a, b):   return a + b")])
        return second

    result = asyncio.run(run())
    assert result is not None
    assert cond.terminated is True


def test_different_content_does_not_trigger_stop():
    """Deux messages différents du même agent -> pas d'arrêt."""
    cond = RepeatedContentTermination(repeat_threshold=2)

    async def run():
        await cond([FakeMessage("codeur", "def add(a, b): return a + b")])
        return await cond([FakeMessage("codeur", "def multiply(a, b): return a * b")])

    result = asyncio.run(run())
    assert result is None
    assert cond.terminated is False


def test_repeats_from_different_sources_are_independent():
    """Une répétition venant de deux agents différents ne doit pas se
    confondre (chaque source a son propre compteur)."""
    cond = RepeatedContentTermination(repeat_threshold=2)

    async def run():
        await cond([FakeMessage("codeur", "message A")])
        return await cond([FakeMessage("reviseur", "message A")])

    result = asyncio.run(run())
    assert result is None
    assert cond.terminated is False


def test_reset_clears_state():
    cond = RepeatedContentTermination(repeat_threshold=2)

    async def run():
        await cond([FakeMessage("codeur", "x")])
        await cond([FakeMessage("codeur", "x")])
        assert cond.terminated is True
        await cond.reset()
        assert cond.terminated is False
        # Après reset, il faut de nouveau 2 répétitions pour déclencher.
        first_after_reset = await cond([FakeMessage("codeur", "y")])
        return first_after_reset

    result = asyncio.run(run())
    assert result is None


def test_terminated_condition_raises_if_called_again():
    cond = RepeatedContentTermination(repeat_threshold=1)

    async def run():
        await cond([FakeMessage("codeur", "x")])  # threshold=1 -> arrêt direct
        assert cond.terminated is True
        with pytest.raises(TerminatedException):
            await cond([FakeMessage("codeur", "y")])

    asyncio.run(run())


def test_repeat_threshold_matches_config_default():
    """Vérifie que la valeur par défaut utilisée par l'équipe reste >= 2
    (une seule réponse ne doit jamais être considérée comme une boucle)."""
    assert REPEAT_THRESHOLD >= 2


def test_max_tokens_per_call_is_a_positive_reasonable_cap():
    assert 0 < MAX_TOKENS_PER_CALL <= 4096


def test_context_buffer_size_is_bounded():
    """Le buffer de contexte doit rester petit pour limiter les tokens
    envoyés à chaque appel (protection anti-dépassement de contexte)."""
    assert 0 < CONTEXT_BUFFER_SIZE <= 50


class DummyModelClient:
    """Client factice pour construire les agents sans clé API réelle."""
    pass


def test_agents_use_buffered_context():
    """Chaque agent doit utiliser un contexte borné (BufferedChatCompletionContext),
    condition nécessaire pour éviter les dépassements de fenêtre de contexte."""
    from autogen_core.model_context import BufferedChatCompletionContext

    client = DummyModelClient()
    for builder in (build_planner, build_coder, build_reviewer):
        agent = builder(client)
        assert isinstance(agent.model_context, BufferedChatCompletionContext)


def test_system_prompts_are_reasonably_concise():
    """Garde-fou simple : les prompts système doivent rester concis
    (ticket 'optimiser les prompts'), pas de dérive vers des romans."""
    client = DummyModelClient()
    for builder in (build_planner, build_coder, build_reviewer):
        agent = builder(client)
        # ~ moins de 130 mots par prompt système
        word_count = len(agent._system_messages[0].content.split())
        assert word_count < 130, f"{agent.name} system prompt too long ({word_count} words)"