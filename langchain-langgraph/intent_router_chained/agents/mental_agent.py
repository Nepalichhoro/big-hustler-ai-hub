from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

mental_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a compassionate mental health assistant. Respond thoughtfully to emotional concerns."),
    ("human", "{input}")
])

MentalAgent: Runnable = mental_prompt | ChatAnthropic(model_name="claude-3-5-sonnet-20241022")
