from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .state import BreakfastState
from .nodes import (
    boil_water,
    toast_bread,
    serve,
    check_toast,
    remember_breakfast
)
from .fix_toast_subgraph import build_fix_toast_subgraph


def build_graph():
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    builder = StateGraph(BreakfastState)

    # Main nodes
    builder.add_node("boil_water", boil_water)
    builder.add_node("toast_bread", toast_bread)
    builder.add_node("check_toast", check_toast)
    builder.add_node("serve", serve)
    builder.add_node("remember_breakfast", remember_breakfast)

    # Subgraph
    fix_toast_subgraph = build_fix_toast_subgraph()
    builder.add_node("fix_toast_subgraph", fix_toast_subgraph)

    # Parallel from START
    builder.add_edge(START, "boil_water")
    builder.add_edge(START, "toast_bread")

    # After parallel → check toast
    builder.add_edge("boil_water", "check_toast")
    builder.add_edge("toast_bread", "check_toast")

    # Conditional routing: check_toast returns name of next node
    builder.add_conditional_edges("check_toast")

    # After serve → remember → END
    builder.add_edge("serve", "remember_breakfast")
    builder.add_edge("remember_breakfast", END)

    graph = builder.compile(checkpointer=checkpointer, store=store)

    return graph, store, checkpointer
