# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 21:40
# @Author  : HeJwei
# @FileName: demo.py
from operator import itemgetter

# Load blog
import bs4
from langchain_community.vectorstores import Chroma
from  my_util.util import QianfanOpenAIEmbeddings, llm
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import os
from langchain.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader


loader = PyPDFLoader("../docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200,
    chunk_overlap=10)

splits = text_splitter.split_documents(documents)
print(len(splits), len(splits[0].page_content))

# Index
batch_size = 15
num_batches = (len(splits) + batch_size - 1) // batch_size  # 计算批次数

vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings())

for i in range(1, num_batches):
    batch = splits[i * batch_size : (i + 1) * batch_size]
    vectorstore.add_documents(documents=batch)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# prompt
class PromptRunnable(Runnable):
    def __init__(self, template):
        self.template = template
    def invoke(self, input, config = None):
        if isinstance(input, dict):
            return self.template.format(**input)
        else:
            raise ValueError("Input should be a dictionary with 'context' and 'question' keys")

qa_template = """您是一个人工智能语言模型助手。您的任务是为给定的用户问题生成五个语义相似的不同版本，以便从向量数据库中检索相关文档。
            通过从多个角度生成用户问题，您的目标是帮助用户克服基于距离的相似性搜索的一些局限性。请将这些替代性问题用换行符\n分隔。
            原始问题：: {question}"""
prompt_perspectives = PromptRunnable(qa_template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [i for i in x.split("\n") if i ])

)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

question = "城院的教学检查要的工作要求是什么"
out = generate_queries.invoke({"question":question})
print(out)
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# docs = retrieval_chain.invoke({"question":question})
# RAG
# RAG
rag_template = """根据这个上下文背景信息回答下列问题:

{context}

问题是: {question}
"""
prompt = PromptRunnable(rag_template)
final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

out = final_rag_chain.invoke({"question":question})
print(out)