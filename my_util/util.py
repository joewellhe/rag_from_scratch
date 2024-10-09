# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 14:10
# @Author  : HeJwei
# @FileName: embeding.py
# 假设这是你自己的嵌入模型，它有一个方法 `embed_texts` 来生成嵌入
import qianfan
from chromadb import Embeddings
import os
from langchain.schema.runnable import Runnable

os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"


# 创建自定义的 OpenAIEmbeddings 类
class QianfanOpenAIEmbeddings(Embeddings):
    def __init__(self):
        self.model = qianfan.Embedding()

    def embed_documents(self, texts):
        # 使用你自己的模型来生成嵌入向量
        resp = self.model.do(
            model="Embedding-V1",
            texts=texts)
        vectors = [data["embedding"] for data in resp["body"]["data"]]
        return vectors

    def embed_query(self, text):
        # 单个文本的嵌入，通常在查询中使用
        return self.embed_documents([text])[0]


class PromptRunnable(Runnable):
    def __init__(self, template):
        self.template = template
    def invoke(self, input, config = None):
        if isinstance(input, dict):
            return self.template.format(**input)
        else:
            raise ValueError("Input should be a dictionary with 'context' and 'question' keys")


chat_comp = qianfan.ChatCompletion()
def llm(question):
    return chat_comp.do(
        model="ERNIE-3.5-8K",
        messages=[{"role": "user", "content": question}]
    )['result']