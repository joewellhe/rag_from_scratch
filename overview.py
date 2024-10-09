# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 15:43
# @Author  : HeJwei
# @FileName: overview.py
import bs4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import qianfan
import os
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough


from chromadb.types import Vector
from langchain_core.output_parsers import StrOutputParser


os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"

# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# print(docs)


# 假设这是你自己的嵌入模型，它有一个方法 `embed_texts` 来生成嵌入
class QianfanEmbeddingModel:

    def __init__(self):
        self.emb = qianfan.Embedding()

    def embed_texts(self, texts):
        resp = self.emb.do(
            model="Embedding-V1",
            texts=texts)
        vectors = [data["embedding"] for data in resp["body"]["data"]]
        return vectors

# 创建自定义的 OpenAIEmbeddings 类
class QianfanOpenAIEmbeddings(Embeddings):
    def __init__(self, model: QianfanEmbeddingModel):
        self.model = model

    def embed_documents(self, texts):
        # 使用你自己的模型来生成嵌入向量
        return self.model.embed_texts(texts)

    def embed_query(self, text):
        # 单个文本的嵌入，通常在查询中使用
        return self.model.embed_texts([text])[0]



loader = PyPDFLoader("docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Embed
# we cant use OpenAIEmbeddings so we should design a QianFanEmbedding
embed_model = QianfanEmbeddingModel()
vectorstore = Chroma.from_documents(documents=splits, embedding=QianfanOpenAIEmbeddings(embed_model))

retriever = vectorstore.as_retriever()

# Prompt
# prompt = ChatPromptTemplate.from_template(template)
# 创建Prompt的Runnable
class PromptRunnable(Runnable):
    def invoke(self, input, config = None):
        if isinstance(input, dict):
            return "请你作为一个知识专家，根据提供的背景信息回答问题，背景信息是'''"+ input['context'] +\
                "'''请你根据以上的背景信息，回答问题: " + input['question']
        else:
            raise ValueError("Input should be a dictionary with 'context' and 'question' keys")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# llm
os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"
chat_comp = qianfan.ChatCompletion()
def llm(question):
    return chat_comp.do(
        model="ERNIE-3.5-8K",
        messages=[{"role": "user", "content": question}]
    )['result']


os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"
# Chain
rag_chain = ( {"context":  retriever | format_docs, "question": RunnablePassthrough()}
              | PromptRunnable()
              | llm)

print(rag_chain.invoke("城市学院的教学评级是啥"))
