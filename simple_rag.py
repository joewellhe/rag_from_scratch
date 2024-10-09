# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 14:15
# @Author  : HeJwei
# @FileName: simple_rag.py
import os
import qianfan
import uuid
from langsmith import traceable

@traceable(run_type="retriever")
def retriever(query: str):
    results = ["秦涛在湖南城市学院学计算机", "他的女朋友是同班同学"]
    return results

# 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk，如何获取请查看https://cloud.baidu.com/doc/Reference/s/9jwvz2egb
os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"
chat_comp = qianfan.ChatCompletion()

# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
@traceable(metadata={"llm": "ERNIE-8K"})
def rag(question):
    docs = retriever(question)
    system_message = """根据以下背景信息回答用户的问题,你不需要提供额外的背景知识,所有回答都依据给定的背景信息,背景信息如下:

    '''{docs}''' the question is: """.format(docs="\n".join(docs))

    return chat_comp.do(
        model="ERNIE-3.5-8K",
        messages=[{"role": "user", "content": system_message+question}]
    )['result']

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langsmith import Client
run_id = str(uuid.uuid4())
rag(
    "秦涛有没有女朋友",
    langsmith_extra={"run_id": run_id, "metadata": {"user_id": "秦涛"}}
)