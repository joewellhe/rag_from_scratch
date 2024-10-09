# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 18:37
# @Author  : HeJwei
# @FileName: rag_fusion.py
from operator import itemgetter

# Load blog
import bs4
from langchain_community.vectorstores import Chroma
from  my_util.util import QianfanOpenAIEmbeddings, llm, PromptRunnable
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import os
from langchain.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# builder a vectorstore
# loader = PyPDFLoader("../docs/937820.pdf")  # for DOCX
# documents = loader.load()
#
# # Split
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=700,
#     chunk_overlap=10)
#
# splits = text_splitter.split_documents(documents)
# print(len(splits), len(splits[0].page_content))
#
# # Index
# batch_size = 15
# num_batches = (len(splits) + batch_size - 1) // batch_size  # 计算批次数
#
# vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings(), persist_directory="../model/vectorstore")
#
# for i in range(1, num_batches):
#     batch = splits[i * batch_size : (i + 1) * batch_size]
#     vectorstore.add_documents(documents=batch)
# vectorstore.persist()
#
vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings(), persist_directory="../model/vectorstore")
retriever = vectorstore.as_retriever()

# prompt
qa_template = """您是一个人工智能语言模型助手。您的任务是为给定的用户问题生成五个语义相似的不同版本，以便从向量数据库中检索相关文档。
            通过从多个角度生成用户问题，您的目标是帮助用户克服基于距离的相似性搜索的一些局限性。请将这些替代性问题用换行符\n分隔。
            原始问题: {question}"""
prompt_perspectives = PromptRunnable(qa_template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [i for i in x.split("\n") if i ])

)

# RAG
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)
    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

question = "抢劫和抢夺的区别是什么"

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})

# RAG


rag_template = """根据这个上下文背景信息回答下列问题:

{context}

问题是: {question}
"""
prompt = PromptRunnable(rag_template)


@traceable
def run_rag_chain(question):
    final_rag_chain = (
            {
                "context": retrieval_chain_rag_fusion,
                "question": itemgetter("question")
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": question})

question = "抢劫是什么，抢劫需要受到什么处罚"
out = run_rag_chain(question)
print(out)
