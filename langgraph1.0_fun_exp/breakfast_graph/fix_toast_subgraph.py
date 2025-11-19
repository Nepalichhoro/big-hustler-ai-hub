from langgraph.graph import StateGraph, END
from .state import BreakfastState


def throw_burnt_bread(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] += ["Threw away burnt bread"]
    return state


def toast_new_bread(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] += ["Toasted new bread"]
    return state


def add_butter(state: BreakfastState) -> BreakfastState:
    state = dict(state)
    state["steps"] += ["Added butter"]
    return state


def build_fix_toast_subgraph():
    builder = StateGraph(BreakfastState)

    builder.add_node("throw_burnt_bread", throw_burnt_bread)
    builder.add_node("toast_new_bread", toast_new_bread)
    builder.add_node("add_butter", add_butter)

    builder.set_entry_point("throw_burnt_bread")
    builder.add_edge("throw_burnt_bread", "toast_new_bread")
    builder.add_edge("toast_new_bread", "add_butter")
    builder.add_edge("add_butter", END)

    return builder.compile()
