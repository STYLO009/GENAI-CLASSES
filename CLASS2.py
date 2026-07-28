from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# LLM
llm = ChatOllama(model='qwen2.5-coder')

# Schema
class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="age of the person")
    city: str = Field(description="city the person belongs to")

# Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Prompt
template = PromptTemplate(
    template="""
Generate a random person's details from {place}.

{format_instructions}
""",
    input_variables=['place'],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)

# Run
prompt = template.invoke({'place': 'India'})
result = llm.invoke(prompt)

# Parse
content = result.content
if isinstance(content, list):
    text_parts = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            text_parts.append(str(item.get("text", item)))
        else:
            text_parts.append(str(item))
    content = "".join(text_parts)
elif not isinstance(content, str):
    content = str(content)

final_result = parser.parse(content)

print(final_result)