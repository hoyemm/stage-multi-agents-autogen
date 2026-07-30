"""
Interface web Streamlit pour l'équipe multi-agents AutoGen.

Lancement :
    streamlit run app_streamlit.py

Fonctionnalités (cf. critères d'acceptation) :
  - Champ de saisie pour la requête utilisateur.
  - Affichage en temps réel des échanges entre agents au fur et à mesure
    qu'ils sont générés (pas d'attente de la fin de la conversation).
  - Bouton pour exporter le résultat final (texte complet de la conversation).
"""

import asyncio
import streamlit as st

# On réutilise telles quelles les fonctions de construction des agents et
# de l'équipe définies dans multi_agent_team.py, pour éviter la duplication
# de logique entre le script CLI et l'interface web.
from multi_agents import (
    build_model_client,
    build_planner,
    build_coder,
    build_reviewer,
    build_docker_executor_agent,
    make_logger,
    SELECTOR_PROMPT,
    MAX_MESSAGES,
    LOG_FILE,
)
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat


st.set_page_config(page_title="Équipe multi-agents AutoGen", page_icon="🤖", layout="centered")
st.title("🤖 Équipe multi-agents — Planificateur / Codeur / Exécuteur / Réviseur")
st.caption(
    "Décrivez votre demande ci-dessous. Les agents collaborent automatiquement "
    "jusqu'à validation du code (exécuté dans un conteneur Docker isolé)."
)

# État persistant entre les interactions Streamlit
if "transcript" not in st.session_state:
    st.session_state.transcript = []  # liste de (source, content)
if "running" not in st.session_state:
    st.session_state.running = False


async def run_team(task: str, placeholder):
    """Exécute l'équipe d'agents et met à jour l'affichage en temps réel."""
    model_client = build_model_client()
    planificateur = build_planner(model_client)
    codeur = build_coder(model_client)
    reviseur = build_reviewer(model_client)
    executeur, docker_executor = await build_docker_executor_agent()

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(MAX_MESSAGES)

    team = SelectorGroupChat(
        [planificateur, codeur, executeur, reviseur],
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )

    log_message = make_logger(LOG_FILE)
    log_message("user", task)
    st.session_state.transcript.append(("user", task))

    try:
        async for message in team.run_stream(task=task):
            if hasattr(message, "source") and hasattr(message, "content"):
                content = str(message.content)
                st.session_state.transcript.append((message.source, content))
                log_message(message.source, content)

                # Ré-affiche tout le transcript à chaque nouveau message
                # (mise à jour "en temps réel" côté interface)
                with placeholder.container():
                    render_transcript()
    finally:
        await docker_executor.stop()
        await model_client.close()


def render_transcript():
    for source, content in st.session_state.transcript:
        if source == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**{source}**")
                st.markdown(content)


# --- Zone de saisie utilisateur -------------------------------------------------
with st.form("request_form", clear_on_submit=False):
    user_task = st.text_area(
        "Votre demande",
        placeholder="Ex : Écris un code Python qui vérifie si un nombre est premier.",
        height=100,
    )
    submitted = st.form_submit_button("Envoyer", disabled=st.session_state.running)

placeholder = st.empty()

# Réaffiche l'historique existant (utile après un rerun Streamlit)
with placeholder.container():
    render_transcript()

if submitted and user_task.strip():
    st.session_state.running = True
    st.session_state.transcript = []  # nouvelle conversation
    asyncio.run(run_team(user_task.strip(), placeholder))
    st.session_state.running = False

# --- Export du résultat ----------------------------------------------------
if st.session_state.transcript:
    full_text = "\n\n".join(
        f"---------- {source} ----------\n{content}"
        for source, content in st.session_state.transcript
    )
    st.download_button(
        label="📥 Exporter le résultat (texte)",
        data=full_text,
        file_name="resultat_conversation.txt",
        mime="text/plain",
    )