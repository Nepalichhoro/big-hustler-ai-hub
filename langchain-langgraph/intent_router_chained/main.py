import os
from dotenv import load_dotenv
from agents.mental_agent import MentalAgent
from agents.nutritional_agent import NutritionalAgent
from router.intent_router import intentRouter

load_dotenv()

async def route_message(user_input: str):
    result = await intentRouter.ainvoke({"input": user_input})

    category = result.content.strip().lower() if isinstance(result.content, str) else None

    if category == "mental":
        response = await MentalAgent.ainvoke({"input": user_input})
        return response.content
    elif category == "nutrition":
        response = await NutritionalAgent.ainvoke({"input": user_input})
        return response.content
    else:
        return "Sorry, I could not determine how to respond to your request."

if __name__ == "__main__":
    import asyncio
    test_input = "I'm feeling anxious and low energy."
    result = asyncio.run(route_message(test_input))
    print(result)
