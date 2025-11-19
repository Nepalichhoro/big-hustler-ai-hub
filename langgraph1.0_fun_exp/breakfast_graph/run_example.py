from langchain_core.runnables import RunnableConfig
from breakfast_graph.graph_builder import build_graph


def main():
    graph, store, checkpointer = build_graph()

    config = {
        "configurable": {
            "thread_id": "breakfast-thread-1",
            "user_id": "user-123",
        }
    }

    print("\n=== Running breakfast graph ===")
    final_state = graph.invoke(
        {"steps": [], "toast_status": "burnt"},
        config=config
    )
    print("Final:", final_state)

    # --- get latest state ---
    snapshot = graph.get_state(config)
    print("\n=== Latest snapshot ===")
    print(snapshot.values)

    # --- get full history ---
    history = list(graph.get_state_history(config))
    print("\n=== Checkpoint History (newest → oldest) ===")
    for i, snap in enumerate(history):
        print(f"Checkpoint {i+1}:")
        print("  step:", snap.metadata.get("step"))
        print("  values:", snap.values)
        print("  checkpoint_id:",
              snap.config["configurable"]["checkpoint_id"])
        print()


if __name__ == "__main__":
    main()
