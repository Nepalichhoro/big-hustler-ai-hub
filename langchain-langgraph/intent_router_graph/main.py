from graph_router import build_router_graph
from dotenv import load_dotenv
load_dotenv()

graph = build_router_graph()

user_input = "I'm feeling anxious and overwhelmed."
state = {"input": user_input}

result = graph.invoke(state)
print("Final Response:", result.get("response"))
