from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .service import get_agent, save_feedback

app = FastAPI(title="GameMatch AI", version=__version__)


class ChatIn(BaseModel):
    session_id: str = Field("default", max_length=100)
    message: str = Field(..., min_length=1, max_length=2000)


class SearchIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    platform: Literal["PC", "PlayStation", "Xbox", "Switch", "Mobile"] | None = None
    budget: Literal["free", "low", "mid", "high"] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    coop: bool | None = None
    max_hours: float | None = Field(None, gt=0)
    k: int = Field(5, ge=1, le=20)


class FeedbackIn(BaseModel):
    session_id: str = Field("default", max_length=100)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field("", max_length=500)


@app.get("/health")
def health() -> dict:
    a = get_agent()
    return {
        "status": "ok", "version": __version__, "games": len(a.catalog.games),
        "graph_engine": a.engine, "llm": a.llm.available,
        "embedder": a.catalog.embedder.name, "index": a.catalog.index.backend,
    }


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    return get_agent().chat(body.session_id, body.message).to_dict()


@app.post("/search")
def search(body: SearchIn) -> dict:
    results = get_agent().search(
        body.query, platform=body.platform, budget=body.budget, difficulty=body.difficulty,
        coop=body.coop, max_hours=body.max_hours, k=body.k,
    )
    return {"results": results}


@app.get("/games/{game_id}")
def game(game_id: str) -> dict:
    g = get_agent().catalog.by_id.get(game_id)
    if g is None:
        raise HTTPException(status_code=404, detail="Игра не найдена")
    return g.to_dict()


@app.post("/feedback")
def feedback(body: FeedbackIn) -> dict:
    save_feedback(body.session_id, body.rating, body.comment)
    return {"status": "saved"}


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str) -> dict:
    get_agent().sessions.pop(session_id, None)
    return {"status": "reset"}
