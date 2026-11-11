from __future__ import annotations
import os
import uuid
import streamlit as st

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="GameMatch AI", page_icon="🎮", layout="wide")

def _local():
    from gamematch.service import get_agent
    return get_agent()

def ask(message: str) -> dict:
    sid = st.session_state.sid
    if API_URL:
        import requests
        r = requests.post(f"{API_URL}/chat", json={"session_id": sid, "message": message}, timeout=60)
        r.raise_for_status()
        return r.json()
    return _local().chat(sid, message).to_dict()

def send_feedback(rating: int, comment: str) -> None:
    sid = st.session_state.sid
    if API_URL:
        import requests
        requests.post(f"{API_URL}/feedback", json={"session_id": sid, "rating": rating, "comment": comment}, timeout=10)
    else:
        from gamematch.service import save_feedback
        save_feedback(sid, rating, comment)

def reset() -> None:
    st.session_state.sid = uuid.uuid4().hex
    st.session_state.messages = []

if "sid" not in st.session_state:
    reset()

st.title("🎮 GameMatch AI")
st.caption("Опишите игру своими словами — например: «хочу атмосферное, как Hollow Knight, но короче».")

with st.sidebar:
    st.button("Новый диалог", on_click=reset, use_container_width=True)
    st.markdown("### Оцените подборку")
    rating = st.radio("Оценка", [5, 4, 3, 2, 1], horizontal=True, label_visibility="collapsed")
    comment = st.text_input("Комментарий (необязательно)")
    if st.button("Отправить оценку", use_container_width=True):
        send_feedback(rating, comment)
        st.success("Спасибо!")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for g in m.get("results", []):
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    appid = g.get("steam_appid")
                    if appid:
                        st.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg", use_column_width=True)
                    else:
                        st.info("Обложка недоступна")
                with col2:
                    st.subheader(f"{g['title']} ({g['year']})")
                    st.caption(f"Платформы: {', '.join(g['platforms'])} · Оценка: {g['rating']:g}/100 · ~{g['hours']:g} ч")
                    st.write(f"**Почему подходит:** {g['why']}")
                    with st.expander("Подробное описание"):
                        st.write(g["description"])

if prompt := st.chat_input("Какую игру ищете?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Ищу…"):
            try:
                data = ask(prompt)
            except Exception as exc:
                data = {"reply": f"Не получилось получить ответ: {exc}", "results": []}
        st.markdown(data["reply"])
        for g in data.get("results", []):
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    appid = g.get("steam_appid")
                    if appid:
                        st.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg", use_column_width=True)
                    else:
                        st.info("Обложка недоступна")
                with col2:
                    st.subheader(f"{g['title']} ({g['year']})")
                    st.caption(f"Платформы: {', '.join(g['platforms'])} · Оценка: {g['rating']:g}/100 · ~{g['hours']:g} ч")
                    st.write(f"**Почему подходит:** {g['why']}")
                    with st.expander("Подробное описание"):
                        st.write(g["description"])
    st.session_state.messages.append({"role": "assistant", "content": data["reply"], "results": data.get("results", [])})
