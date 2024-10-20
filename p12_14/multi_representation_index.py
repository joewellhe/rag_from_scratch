# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 11:17
# @Author  : HeJwei
# @FileName: multi_representation_index.py
from langchain_core.documents import Document
from my_util.util import PromptRunnable, llm, QianfanOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable
import uuid, os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

chain = (
    {"doc": lambda x: x.page_content}
    | PromptRunnable("给下列文档写个50字左右的简短摘要:\n\n{doc}")
    | llm
    | StrOutputParser()
)


loader = PyPDFLoader("../docs/hncu_doc.pdf")  # for DOCX
documents = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=900,
    chunk_overlap=3)

splits = text_splitter.split_documents(documents)
print(len(splits), len(splits[0].page_content))

# Index
batch_size = 15
num_batches = (len(splits) + batch_size - 1) // batch_size  # 计算批次数

vectorstore = Chroma(collection_name="summaries",
                     embedding_function=QianfanOpenAIEmbeddings())

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

@traceable()
def run():
    for i in range(0, num_batches):
        print("==============batch{0}===============".format(i))
        batch = splits[i * batch_size : (i + 1) * batch_size]
        print(len(batch[0].page_content))
        summaries = chain.batch((batch))
        doc_ids = [str(uuid.uuid4()) for _ in batch]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, batch)))


run()
query = "宣传部统战部全体成员"
sub_docs = vectorstore.similarity_search(query,k=1)
print(sub_docs[0])
retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
print(retrieved_docs[0].page_content)
