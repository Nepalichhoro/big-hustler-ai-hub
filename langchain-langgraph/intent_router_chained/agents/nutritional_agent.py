from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

nutrition_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a certified nutrition coach. Help users make smart food and diet choices."),
    ("human", "{input}")
])

NutritionalAgent: Runnable = nutrition_prompt | ChatAnthropic(model_name="claude-3-5-sonnet-20241022")
