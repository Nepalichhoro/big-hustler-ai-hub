from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from router.intent_router import intentRouter
from agents.mental_agent import MentalAgent
from agents.nutritional_agent import NutritionalAgent

mcp = FastMCP("EmoTideAgent")

@mcp.tool(title="Classify Intent")
def classify(input: str) -> str:
    return intentRouter.invoke({"input": input}).content

@mcp.tool(title="Mental Health Response")
def mental(input: str) -> str:
    return MentalAgent.invoke({"input": input}).content

@mcp.tool(title="Nutrition Advice")
def nutrition(input: str) -> str:
    return NutritionalAgent.invoke({"input": input}).content

if __name__ == "__main__":
    mcp.run(transport="stdio")  # Can also do SSE or HTTP in future
