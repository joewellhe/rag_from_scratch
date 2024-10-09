# -*- coding: utf-8 -*-
# @Time    : 2024/10/6 19:51
# @Author  : HeJwei
# @FileName: HyDE.py
from my_util.util import PromptRunnable
from my_util.util import llm, QianfanOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langsmith import traceable
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

Hyde_template = """请作为一个法律专家,写一个专业的法律短文本(约150字)回答下面的问题
问题是: {question}
法律短文本:"""

hyde_prompt = PromptRunnable(Hyde_template)

hyde_chain = hyde_prompt | llm | StrOutputParser()


question = "15岁的人故意杀人会受到什么样的处罚?"
# print(hyde_chain.invoke({"question": question}))

vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings(), persist_directory="../model/vectorstore")
retriever = vectorstore.as_retriever()

# Retrieve
retrieval_chain = hyde_chain | retriever

# final chain
rag_template = """根据这个上下文背景信息回答下列问题:

{context}

问题是: {question}
"""
rag_prompt = PromptRunnable(rag_template)

@traceable()
def run_hyde(q):
    final_chain = ({"question": RunnablePassthrough(),
                    "context": retrieval_chain}
                   | rag_prompt
                   | llm
                   | StrOutputParser())
    return final_chain.invoke({"question": q})

print(run_hyde(question))