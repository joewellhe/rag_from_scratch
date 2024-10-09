# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 14:00
# @Author  : HeJwei
# @FileName: index.py

import tiktoken
import numpy as np
from langchain.prompts import ChatPromptTemplate


question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Count tokens considering ~4 char / token
num = num_tokens_from_string(question, "cl100k_base")

# Text embedding models
from my_util.util import QianfanOpenAIEmbeddings
# embed = QianfanOpenAIEmbeddings()
# query_result = embed.embed_query(question)
# document_result = embed.embed_query(document)

# Cosine similarity is reccomended (1 indicates identical) for OpenAI embeddings.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# similarity = cosine_similarity(query_result, document_result)

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)
print(len(splits), len(splits[0].page_content))
# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

batch_size = 15
num_batches = (len(splits) + batch_size - 1) // batch_size  # 计算批次数

vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings())

for i in range(1, num_batches):
    batch = splits[i * batch_size : (i + 1) * batch_size]
    vectorstore.add_documents(documents=batch)

print(len(vectorstore))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("What is Task Decomposition?")
print(len(docs))

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
from langchain.schema.runnable import Runnable, RunnablePassthrough
class PromptRunnable(Runnable):
    def invoke(self, input, config = None):
        if isinstance(input, dict):
            return template.format(**input)
        else:
            raise ValueError("Input should be a dictionary with 'context' and 'question' keys")

from my_util.util import llm
prompt = PromptRunnable()
# chain = prompt | llm
# out = chain.invoke({"context":docs, "question":"What is Task Decomposition?"})
# print(out)

# rag_chain
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
out = rag_chain.invoke("What is Task Decomposition?")
print(out)