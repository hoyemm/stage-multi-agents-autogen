"""
Interface web Streamlit pour l'équipe multi-agents AutoGen.

Lancement :
    streamlit run streamlit_app.py

Fonctionnalités (cf. critères d'acceptation) :
  - Champ de saisie pour la requête utilisateur.
  - Affichage en temps réel des échanges entre agents au fur et à mesure
    qu'ils sont générés (pas d'attente de la fin de la conversation).
  - Bouton pour exporter le résultat final (texte complet de la conversation).
  - Suivi du nombre de tokens consommés (contrôle des coûts), avec arrêt
    automatique si le budget de tokens ou le nombre de messages est dépassé,
    ou si une boucle est détectée entre agents.
"""

import asyncio
import streamlit as st

# On réutilise telles quelles les fonctions de construction des agents et
# de l'équipe définies dans multi_agents.py, pour éviter la duplication
# de logique entre le script CLI et l'interface web.
from multi_agents import (
    build_model_client,
    build_team,
    make_logger,
    extract_usage,
    LOG_FILE,
    MAX_MESSAGES,
    MAX_TOTAL_TOKENS,
)
from autogen_agentchat.messages import StopMessage


st.set_page_config(page_title="Équipe multi-agents AutoGen", page_icon="🤖", layout="centered")
st.title("🤖 Équipe multi-agents — Planificateur / Codeur / Exécuteur / Réviseur")
st.caption(
    "Décrivez votre demande ci-dessous. Les agents collaborent automatiquement "
    "jusqu'à validation du code (exécuté dans un conteneur Docker isolé)."
)
st.caption(
    f"Garde-fous actifs : max {MAX_MESSAGES} messages, "
    f"budget ≈ {MAX_TOTAL_TOKENS} tokens, arrêt automatique si boucle détectée."
)

# État persistant entre les interactions Streamlit
if "transcript" not in st.session_state:
    st.session_state.transcript = []  # liste de (source, content)
if "running" not in st.session_state:
    st.session_state.running = False
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


async def run_team(task: str, placeholder, token_placeholder):
    """Exécute l'équipe d'agents et met à jour l'affichage en temps réel."""
    model_client = build_model_client()
    team, docker_executor = await build_team(model_client)

    log_message = make_logger(LOG_FILE)
    log_message("user", task)
    st.session_state.transcript.append(("user", task))
    st.session_state.total_tokens = 0

    try:
        async for message in team.run_stream(task=task):
            if hasattr(message, "source") and hasattr(message, "content"):
                content = str(message.content)
                st.session_state.transcript.append((message.source, content))

                usage = extract_usage(message)
                log_message(message.source, content, usage)
                if usage and usage.get("prompt_tokens") and usage.get("completion_tokens"):
                    st.session_state.total_tokens += (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                    )

                # Ré-affiche tout le transcript à chaque nouveau message
                # (mise à jour "en temps réel" côté interface)
                with placeholder.container():
                    render_transcript()
                token_placeholder.caption(
                    f"🔢 Tokens consommés (estimé) : {st.session_state.total_tokens}"
                )
            elif isinstance(message, StopMessage):
                # Arrêt anticipé (budget tokens dépassé, boucle détectée, ...)
                st.session_state.transcript.append((message.source, message.content))
                log_message(message.source, message.content)
                with placeholder.container():
                    render_transcript()
                st.warning(f"Conversation arrêtée automatiquement : {message.content}")
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
token_placeholder = st.empty()

# Réaffiche l'historique existant (utile après un rerun Streamlit)
with placeholder.container():
    render_transcript()
if st.session_state.total_tokens:
    token_placeholder.caption(f"🔢 Tokens consommés (estimé) : {st.session_state.total_tokens}")

if submitted and user_task.strip():
    st.session_state.running = True
    st.session_state.transcript = []  # nouvelle conversation
    asyncio.run(run_team(user_task.strip(), placeholder, token_placeholder))
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