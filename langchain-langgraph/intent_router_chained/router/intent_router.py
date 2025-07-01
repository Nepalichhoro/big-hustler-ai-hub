from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's message into one of the following categories:
- "mental" for mental health or emotional concerns
- "nutrition" for food, diet, or health guidance

Return only "mental" or "nutrition"."""),
    ("human", "{input}")
])

intentRouter: Runnable = router_prompt | ChatAnthropic(model_name="claude-3-5-sonnet-20241022")
