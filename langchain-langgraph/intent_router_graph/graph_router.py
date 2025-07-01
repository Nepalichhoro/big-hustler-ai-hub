from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from router.intent_router import intentRouter
from agents.mental_agent import MentalAgent
from agents.nutritional_agent import NutritionalAgent
from typing import TypedDict

# 1. Define schema
class GraphState(TypedDict, total=False):
    input: str
    category: str
    response: str

# 2. Node functions
def classify_intent(state: GraphState) -> GraphState:
    result = intentRouter.invoke({"input": state["input"]})
    category = result.content.strip().lower() if isinstance(result.content, str) else "unknown"
    return {**state, "category": category}

def route_by_category(state: GraphState) -> str:
    if state["category"] == "mental":
        return "mental"
    elif state["category"] == "nutrition":
        return "nutrition"
    return END

def handle_mental(state: GraphState) -> GraphState:
    result = MentalAgent.invoke({"input": state["input"]})
    return {**state, "response": result.content}

def handle_nutrition(state: GraphState) -> GraphState:
    result = NutritionalAgent.invoke({"input": state["input"]})
    return {**state, "response": result.content}

# 3. Build the graph
def build_router_graph() -> Runnable:
    builder = StateGraph(GraphState)  # ✅ pass schema

    builder.add_node("classify", classify_intent)
    builder.add_node("mental", handle_mental)
    builder.add_node("nutrition", handle_nutrition)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_by_category, {
        "mental": "mental",
        "nutrition": "nutrition",
        END: END
    })

    builder.add_edge("mental", END)
    builder.add_edge("nutrition", END)

    return builder.compile()
