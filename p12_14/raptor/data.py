# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 21:12
# @Author  : HeJwei
# @FileName: data.py

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import os
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count():
    loader = PyPDFLoader("../../docs/health_diet.pdf")  # for DOCX
    docs = loader.load()

    print("=======data info======")
    print(len(docs), len(docs[0].page_content))
    print(docs[0].page_content)

    docs_texts = [d.page_content for d in docs]

    # Calculate the number of tokens for each document
    counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Token Counts")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Display the histogram
    plt.show()

    chunk_size_tok = 2000
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=0
    )

    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    print(
        "Num tokens in all context: %s"
        % num_tokens_from_string(concatenated_content, "cl100k_base")
    )
    texts_split = text_splitter.split_text(concatenated_content)
    print("=======texts_split info======")
    print(len(texts_split), len(texts_split[0]))
    print(texts_split[0])

def get_documents():
    loader = PyPDFLoader("../../docs/health_diet.pdf")  # for DOCX
    docs = loader.load()
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    chunk_size_tok = 400
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=0
    )
    texts_split = text_splitter.split_text(concatenated_content)
    return texts_split

x = get_documents()
print(len(x))
max_len = max([len(i) for i in x])
print(max_len)
