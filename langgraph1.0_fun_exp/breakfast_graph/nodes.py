from typing import TypedDict, List
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import uuid

from .state import BreakfastState


def boil_water(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["Boiled water"]
    return state


def toast_bread(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["Toasted bread"]

    # For demo, keep original toast_status unless provided
    state["toast_status"] = state.get("toast_status", "ok")
    return state


def serve(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["Served breakfast"]
    return state


def check_toast(state: BreakfastState) -> str:
    """Routing function for branching."""
    if state["toast_status"] == "burnt":
        return "fix_toast_subgraph"
    return "serve"


def remember_breakfast(
    state: BreakfastState,
    config: RunnableConfig,
    *,
    store: BaseStore
) -> BreakfastState:
    """Store permanent memory about user's breakfast."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "breakfast_history")
    memory_id = str(uuid.uuid4())

    store.put(
        namespace,
        memory_id,
        {
            "steps": state["steps"],
            "toast_status": state["toast_status"],
        },
    )

    return state
