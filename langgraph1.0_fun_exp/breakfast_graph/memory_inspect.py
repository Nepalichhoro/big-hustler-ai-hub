from breakfast_graph.graph_builder import build_graph
from pprint import pprint


def main():
    graph, store, _ = build_graph()

    namespace = ("user-123", "breakfast_history")

    print("\n=== Searching long-term breakfast memory ===")
    items = store.search(namespace, query=None, limit=10)

    for item in items:
        pprint(item.dict())


if __name__ == "__main__":
    main()
