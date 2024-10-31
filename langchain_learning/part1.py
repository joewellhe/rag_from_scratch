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
from langsmith import traceable
import os

from win32comext.adsi.demos.scp import verbose

os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Chat Messages
chat = QianfanChatEndpoint(temperature=.7, model="ERNIE-3.5-8K")
res = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)
print(res)

# Document

doc = Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })

print(doc)

# Language Model
llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", streaming=True, temperature=0.8)
res = llm("What day comes after Friday?")
print(res)

# Chat Model
# A model that takes a series of messages and returns a message output
chat = QianfanChatEndpoint(temperature=.7, model="ERNIE-3.5-8K")
res = chat(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)
print(res)

# Text Embedding Model
embeddings = QianfanEmbeddingsEndpoint(model="Embedding-V1")
text = "Hi! It's time for the beach"
text_embedding = embeddings.embed_query(text)
print (f"Here's a sample: {text_embedding[:5]}...")
print (f"Your embedding is length {len(text_embedding)}")
txt1 = "This is my document. It is full of text that I've gathered from other places"
txt2 = "今天天气很不错"
docs = [txt1, txt2]
embeds = embeddings.embed_documents(docs)
print(embeds)

# Prompt Template
template  = '''
I really want to travel to {location}. What should I do there?

Respond in one short sentence
'''
prompt = PromptTemplate(input_variables=['location'],
                        template=template)
print(prompt.format(location="ROMA"))

# Example Selectors
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\n Example Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]
embeddings = QianfanEmbeddingsEndpoint(model="Embedding-V1")
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Chroma,
    k=2
)

similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    # Your prompt
    example_prompt=example_prompt,
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\nOutput:",
    # What inputs your prompt will receive
    input_variables=["noun"],
)


# Select a noun!
my_noun = "student"
# my_noun = "student"
prompt = similar_prompt.format(noun=my_noun)
print(prompt)
llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.8)
print(llm(prompt))

# Output Parsers
# method 1 Prompt Instructions & String Parsing
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.8)
# How you would like your response structured. This is basically a fancy prompt template
response_scheme = [
    ResponseSchema(name="bad string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# See the prompt template you created for formatting
output_parser = StructuredOutputParser.from_response_schemas(response_scheme)
format_instructions = output_parser.get_format_instructions()
template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)
prompt_value = prompt.format(user_input="welcom to califonya!")
print(prompt_value)
print(llm(prompt_value))

# Output Parsers Method 2: OpenAI Fuctions
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_chain
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Optional, List


class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


class PeopleList(BaseModel):
    """List of people"""
    people: List[Person]

# 创建输出解析器
parser = PydanticOutputParser(pydantic_object=PeopleList)
print(parser.get_format_instructions())
# 创建提示模板
prompt = ChatPromptTemplate.from_template("""
Extract information about people from the following text. Include their name, age if mentioned,
and anything they are described as liking.

Text: {input}

{format_instructions}

Important:
- Return pure JSON without any markdown code blocks
- Do not include any comments
- Use null for missing information
- For age calculations, please calculate the actual number
""")

# 使用 QianfanChatEndpoint 而不是 QianfanLLMEndpoint
llm = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.1
)

# 组合提示模板和输出解析器
chain = prompt | llm | parser

# 运行链
result = chain.invoke({
    "input": "张三13岁, 王五18岁爱打篮球. 刘星5岁爱吃冰淇淋.",
    "format_instructions": parser.get_format_instructions()
})

print(result)

# Indexes - Structuring documents to LLMs can work with them
# Document Loaders
from langchain.document_loaders import HNLoader
loader = HNLoader("https://news.ycombinator.com/item?id=34422627")
data = loader.load()
print(f"Found {len(data)} comments")
print (f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()

print (f"You have {len(documents)} document")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.split_documents(documents)
print (f"You have {len(texts)} documents")
print ("Preview:")
print (texts[0].page_content, "\n")
print (texts[1].page_content)

# Retrievers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()
print (f"Initially, You have {len(documents)} document")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.split_documents(documents)
embeddings = QianfanEmbeddingsEndpoint(model="Embedding-V1")
print (f"You have {len(texts)} documents")
# 分批次增加vectorstore
vectorstore = Chroma(embedding_function=embeddings)

for i in range(0, len(texts), 12):
    batch = texts[i:i+12]
    vectorstore.add_documents(documents=batch)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("专业评估认证是提升专业办学质量、提高学校教育教学水平")
print(docs)


# Chat Message History
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("hi")
history.add_ai_message("hi")
history.add_user_message("what is the capital of france?")
print(history)
chat = QianfanChatEndpoint(temperature=.7, model="ERNIE-3.5-8K")
ai_response = chat(history.messages)
print(ai_response)
history.add_ai_message(ai_response.content)
print(history)

# Chains
from langchain.chains import SimpleSequentialChain, LLMChain

@traceable()
def run():
    template = """Your job is to come up with a classic dish from the area that the users suggests.
    % USER LOCATION
    {user_location}

    YOUR RESPONSE:
    """

    prompt_template = PromptTemplate(input_variables=["user_location"], template=template)
    llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.1)
    location_chain = LLMChain(llm=llm, prompt=prompt_template)
    template = """Given a meal, give a short and simple recipe on how to make that dish at home.
    % MEAL
    {user_meal}

    YOUR RESPONSE:
    """
    prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)
    meal_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Given a English document, please translate it to Chinese.
    % MEAL
    {document}

    YOUR RESPONSE:
    """
    prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)
    trans_chain = LLMChain(llm=llm, prompt=prompt_template)
    overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain, trans_chain], verbose=True)
    review = overall_chain.run("北京")
    print(review)

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()
print (f"Initially, You have {len(documents)} document")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=10)

# Split your docs into texts
texts = text_splitter.split_documents(documents)
llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", temperature=0.1)

@traceable()
def run():
    # There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.run(texts)

run()
