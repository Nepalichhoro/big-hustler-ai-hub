from typing import TypedDict, List

class BreakfastState(TypedDict):
    steps: List[str]
    toast_status: str  # "ok" or "burnt"
