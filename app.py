import os
import asyncio
import random
from typing import Dict, List, TypedDict, Annotated
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_core.documents import Document
import networkx as nx
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── LLM: Gemini ───────────────────────────────────────────────────────────────
# Твой ключ вставлен сюда
GEMINI_API_KEY = "AIzaSyBWm2r8yM8TV2szvPgUFJZ2yNKa_-XslRg"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",          # или "gemini-1.5-pro" если доступ открыт
    temperature=0.7,
    google_api_key=GEMINI_API_KEY
)

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(simulate())
    yield

# ─── Embeddings для векторной БД (тоже Gemini) ────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_document"
)

# Векторная БД
vector_db = Chroma(collection_name="agent_memories", embedding_function=embeddings)

# ─── Остальная логика остаётся почти без изменений ────────────────────────────

EMOTIONS = ["happy", "sad", "angry", "neutral"]

def update_emotion(current: str, event: str) -> str:
    event_lower = event.lower()
    if any(w in event_lower for w in ["good", "great", "happy", "nice", "love", "positive", "клад", "нашёл"]):
        return "happy"
    if any(w in event_lower for w in ["bad", "sorry", "fail", "angry", "hate", "negative"]):
        return random.choice(["sad", "angry"])
    return current

def emotion_influenced_prompt(base_prompt: str, emotion: str) -> str:
    style = ""
    if emotion == "happy":
        style = " Отвечай весело, позитивно, с восклицаниями и смайликами :)"
    elif emotion == "sad":
        style = " Отвечай грустно, меланхолично, с ноткой печали..."
    elif emotion == "angry":
        style = " Отвечай раздражённо, резко, можешь использовать CAPS и восклицательные знаки!!!"
    return base_prompt + style

# Граф отношений
relations_graph = nx.Graph()

def update_relationship(agent1: str, agent2: str, sympathy: int):
    relations_graph.add_edge(agent1, agent2, weight=sympathy)

# Состояние агента и глобальное
class AgentState(TypedDict):
    name: str
    personality: str
    mood: str
    memories: List[str]
    plans: str
    relations: Dict[str, int]

class GlobalState(TypedDict):
    agents: Dict[str, AgentState]
    events: Annotated[List[str], lambda x, y: x + y]
    user_input: str

# ─── Ноды графа ────────────────────────────────────────────────────────────────
async def reflect(state: GlobalState):
    for name, agent in state["agents"].items():
        prompt = ChatPromptTemplate.from_template(
            """Ты — {name}. Твоя личность: {personality}.
Текущее настроение: {mood}.
Ключевые воспоминания: {memories}.
Недавние события: {events}.

Проанализируй ситуацию и поставь себе новую цель/план на следующие действия."""
        )
        chain = prompt | llm | StrOutputParser()
        response = await chain.ainvoke({
            "name": name,
            "personality": agent["personality"],
            "mood": agent["mood"],
            "memories": "\n".join(agent["memories"][-5:]),
            "events": "\n".join(state["events"][-5:])
        })
        agent["plans"] = response.strip()
    return state

async def act(state: GlobalState):
    new_events = []
    agent_names = list(state["agents"].keys())

    for name, agent in state["agents"].items():
        if len(agent_names) < 2:
            continue
        other = random.choice([n for n in agent_names if n != name])

        base_prompt = f"""Ты — {name}. Сейчас ты общаешься с {other}.
Твой план: {agent['plans']}.
Отношения с ним: {agent['relations'].get(other, 0)} (чем выше — тем лучше).

Напиши короткое сообщение, которое ты хочешь сказать {other}."""
        
        full_prompt = emotion_influenced_prompt(base_prompt, agent["mood"])
        prompt = ChatPromptTemplate.from_template(full_prompt)
        chain = prompt | llm | StrOutputParser()
        message = (await chain.ainvoke({})).strip()

        # Изменение отношения
        sympathy_change = random.randint(-8, 12) if "love" in message.lower() or "друг" in message.lower() else random.randint(-10, 10)
        old_sym = agent["relations"].get(other, 0)
        new_sym = max(-100, min(100, old_sym + sympathy_change))
        agent["relations"][other] = new_sym
        update_relationship(name, other, new_sym)

        # Обновление настроения
        event_text = f"{name} → {other}: {message}"
        agent["mood"] = update_emotion(agent["mood"], message)

        # Память
        doc = Document(page_content=event_text)
        vector_db.add_documents([doc])
        agent["memories"].append(event_text)

        # Суммаризация, если слишком много
        if len(agent["memories"]) > 12:
            sum_prompt = ChatPromptTemplate.from_template(
                "Суммаризируй эти воспоминания в 3–5 предложений, сохранив самое важное:\n{mem}"
            )
            summary_chain = sum_prompt | llm | StrOutputParser()
            summary = (await summary_chain.ainvoke({"mem": "\n".join(agent["memories"])})).strip()
            agent["memories"] = [summary] + agent["memories"][-3:]

        new_events.append(event_text)

    state["events"].extend(new_events)
    return state

# ─── LangGraph ─────────────────────────────────────────────────────────────────
workflow = StateGraph(GlobalState)
workflow.add_node("reflect", reflect)
workflow.add_node("act", act)
workflow.add_edge(START, "reflect")
workflow.add_edge("reflect", "act")
workflow.add_edge("act", END)

graph = workflow.compile()

# Начальные агенты (можно добавить больше)
global_state = {
    "agents": {
        "Алекс": {"name": "Алекс", "personality": "Дружелюбный исследователь", "mood": "neutral", "memories": [], "plans": "", "relations": {}},
        "Мария": {"name": "Мария", "personality": "Осторожный аналитик", "mood": "neutral", "memories": [], "plans": "", "relations": {}},
        "Виктор": {"name": "Виктор", "personality": "Авантюрный творец", "mood": "neutral", "memories": [], "plans": "", "relations": {}},
    },
    "events": [],
    "user_input": ""
}

# ─── Симуляция в фоне ──────────────────────────────────────────────────────────
async def simulate():
    global global_state
    while True:
        global_state = await graph.ainvoke(global_state)
        await broadcast_state()
        await asyncio.sleep(12)  # можно регулировать слайдером на фронте

# WebSocket
connections: List[WebSocket] = []

async def broadcast_state():
    state_data = {
        "events": global_state["events"][-15:],
        "agents": global_state["agents"],
        "graph": [{"source": u, "target": v, "weight": d["weight"]} for u, v, d in relations_graph.edges(data=True)]
    }
    for ws in connections[:]:
        try:
            await ws.send_json(state_data)
        except:
            connections.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            global_state["events"].append(f"[Игрок]: {data}")
            global_state = await graph.ainvoke(global_state)
            await broadcast_state()
    except WebSocketDisconnect:
        connections.remove(websocket)

# API
@app.get("/agents/{name}")
def get_agent(name: str):
    return global_state["agents"].get(name, {})

@app.post("/event")
async def add_event(event: Dict):
    text = event.get("text", "")
    global_state["events"].append(f"[Событие]: {text}")
    global_state = await graph.ainvoke(global_state)
    return {"status": "added"}

# ... весь остальной код выше (app = FastAPI(), другие эндпоинты и т.д.)

@app.get("/graph")
def get_graph():
    return [{"source": u, "target": v, "weight": d["weight"]} 
            for u, v, d in relations_graph.edges(data=True)]

# ← Корневой маршрут — тоже на верхнем уровне
@app.get("/")
def root():
    return {
        "message": "КИБЕР РЫВОК хакатон 2026 — сервер работает!",
        "docs": "Перейди сюда: /docs",
        "status": "online",
        "time": "симуляция агентов запущена"
    }

# Рекомендую перейти на lifespan вместо устаревшего on_event
# (это убирает DeprecationWarning)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(simulate())
    yield
    # Shutdown (если нужно что-то почистить)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)