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
results = llm.invoke(prompt)

# Parse
final_result = parser.parse(results.content)

print(final_result)