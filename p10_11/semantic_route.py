# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 14:32
# @Author  : HeJwei
# @FileName: semantic_route.py
from my_util.util import QianfanOpenAIEmbeddings, PromptRunnable, llm
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langsmith import traceable
import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""


# Embed prompts
embeddings = QianfanOpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)
print(prompt_embeddings)


# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptRunnable(most_similar)

@traceable()
def semantic_router(input):
    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | llm
        | StrOutputParser()
    )

    return chain.invoke(input)

print(semantic_router("What's a black hole"))

