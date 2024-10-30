from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.llms import QianfanLLMEndpoint
from langchain.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain.schema import Document
from langchain import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
import os


os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"

# # Chat Messages
# chat = QianfanChatEndpoint(temperature=.7, model="ERNIE-3.5-8K")
# res = chat(
#     [
#         SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
#         HumanMessage(content="I like the beaches where should I go?"),
#         AIMessage(content="You should go to Nice, France"),
#         HumanMessage(content="What else should I do when I'm there?")
#     ]
# )
# print(res)
#
# # Document
#
# doc = Document(page_content="This is my document. It is full of text that I've gathered from other places",
#          metadata={
#              'my_document_id' : 234234,
#              'my_document_source' : "The LangChain Papers",
#              'my_document_create_time' : 1680013019
#          })
#
# print(doc)
#
# # Language Model
# llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", streaming=True, temperature=0.8)
# res = llm("What day comes after Friday?")
# print(res)
#
# # Chat Model
# # A model that takes a series of messages and returns a message output
# chat = QianfanChatEndpoint(temperature=.7, model="ERNIE-3.5-8K")
# res = chat(
#     [
#         SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
#         HumanMessage(content="I would like to go to New York, how should I do this?")
#     ]
# )
# print(res)
#
# # Text Embedding Model
# embeddings = QianfanEmbeddingsEndpoint(model="Embedding-V1")
# text = "Hi! It's time for the beach"
# text_embedding = embeddings.embed_query(text)
# print (f"Here's a sample: {text_embedding[:5]}...")
# print (f"Your embedding is length {len(text_embedding)}")
# txt1 = "This is my document. It is full of text that I've gathered from other places"
# txt2 = "今天天气很不错"
# docs = [txt1, txt2]
# embeds = embeddings.embed_documents(docs)
# print(embeds)
#
# Prompt Template
# template  = '''
# I really want to travel to {location}. What should I do there?
#
# Respond in one short sentence
# '''
# prompt = PromptTemplate(input_variables=['location'],
#                         template=template)
# print(prompt.format(location="ROMA"))

# # Example Selectors
# example_prompt = PromptTemplate(
#     input_variables=["input", "output"],
#     template="Example Input: {input}\n Example Output: {output}",
# )
#
# # Examples of locations that nouns are found
# examples = [
#     {"input": "pirate", "output": "ship"},
#     {"input": "pilot", "output": "plane"},
#     {"input": "driver", "output": "car"},
#     {"input": "tree", "output": "ground"},
#     {"input": "bird", "output": "nest"},
# ]
# embeddings = QianfanEmbeddingsEndpoint(model="Embedding-V1")
# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples,
#     embeddings,
#     Chroma,
#     k=2
# )
#
# similar_prompt = FewShotPromptTemplate(
#     # The object that will help select examples
#     example_selector=example_selector,
#     # Your prompt
#     example_prompt=example_prompt,
#     # Customizations that will be added to the top and bottom of your prompt
#     prefix="Give the location an item is usually found in",
#     suffix="Input: {noun}\nOutput:",
#     # What inputs your prompt will receive
#     input_variables=["noun"],
# )
#
#
# # Select a noun!
# my_noun = "student"
# # my_noun = "student"
# prompt = similar_prompt.format(noun=my_noun)
# print(prompt)
# llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.8)
# print(llm(prompt))

# Output Parsers
# method 1 Prompt Instructions & String Parsing
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.8)
# # How you would like your response structured. This is basically a fancy prompt template
# response_scheme = [
#     ResponseSchema(name="bad string", description="This a poorly formatted user input string"),
#     ResponseSchema(name="good_string", description="This is your response, a reformatted response")
# ]
#
# # See the prompt template you created for formatting
# output_parser = StructuredOutputParser.from_response_schemas(response_scheme)
# format_instructions = output_parser.get_format_instructions()
# template = """
# You will be given a poorly formatted string from a user.
# Reformat it and make sure all the words are spelled correctly
#
# {format_instructions}
#
# % USER INPUT:
# {user_input}
#
# YOUR RESPONSE:
# """
# prompt = PromptTemplate(
#     input_variables=["user_input"],
#     partial_variables={"format_instructions": format_instructions},
#     template=template
# )
# prompt_value = prompt.format(user_input="welcom to califonya!")
# print(prompt_value)
# print(llm(prompt_value))

# Output Parsers Method 2: OpenAI Fuctions
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_chain
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

# 创建输出解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""
Extract information about people from the following text. Include their name, age if mentioned, 
and anything they are described as liking.

Text: {input}

{format_instructions}

Remember to extract all people mentioned in the text, even if some information is missing for them.
Just mark missing information as null.
""")

# 使用 QianfanChatEndpoint 而不是 QianfanLLMEndpoint
llm = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.8
)

# 组合提示模板和输出解析器
chain = prompt | llm | parser

# 运行链
result = chain.invoke({
    "input": "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally.",
    "format_instructions": parser.get_format_instructions()
})