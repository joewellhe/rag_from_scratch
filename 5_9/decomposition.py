# -*- coding: utf-8 -*-
# @Time    : 2024/10/3 16:16
# @Author  : HeJwei
# @FileName: deposition.py

# Decomposition
import os
from my_util.util import PromptRunnable
from my_util.util import llm, QianfanOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langsmith import traceable
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

decomposition_template = """你是一位有帮助的助手，能够根据输入的问题生成多个子问题。目标是将输入的问题拆解为一组可以单独回答的子问题
子问题生成与以下内容相关的多个搜索查询： {question} 输出 (3 子问题), 每个子问题用换行符\n分隔:"""

prompt_decomposition = PromptRunnable(decomposition_template)
# Chain
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: [xi for xi in x.split("\n") if xi]))

# Run
question = "15岁的人故意杀人会受到什么样的处罚?"
# questions = generate_queries_decomposition.invoke({"question":question})
# print(questions)

# Answer recursively
# Prompt
recursively_template = """这是你需要回答的问题:

\n --- \n {question} \n --- \n

这是可用的背景问题和它的答案:

\n --- \n {q_a_pairs} \n --- \n

这是额外的与该需要回答的问题相关的上下文信息: 

\n --- \n {context} \n --- \n

使用以上上下文信息和背景信息以及其答案回答这个问题: \n {question}
"""
recursively_prompt = PromptRunnable(recursively_template)


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings(), persist_directory="../model/vectorstore")
retriever = vectorstore.as_retriever()


@traceable
def run_recursively_decomposition(question):
    answer = None
    q_a_pairs = ""
    questions = generate_queries_decomposition.invoke({"question": question})
    for q in questions:
        rag_chain = (
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question"),
                 "q_a_pairs": itemgetter("q_a_pairs")}
                | recursively_prompt
                | llm
                | StrOutputParser())

        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

    return answer

# res = run_recursively_decomposition(question)
# print(res)

# Answer individually
from langchain import hub

rag_template = '''
你是一个回答QA问答任务的助手, 使用一下所检索到的上下文来回答问题, 如果你不知道如何回答这个问题
你只需要说不知道, 请最多使用不超过3个句子回答问题,并且保持回答的准确和简洁
\nQuestion: {question} \nContext: {context} \nAnswer:
'''
rag_prompt = PromptRunnable(rag_template)

def individually_decomposition(question, prompt_rag):
    """RAG on each sub-question"""

    # Use our decomposition /
    sub_questions = generate_queries_decomposition.invoke({"question": question})
    print(sub_questions)
    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs,
                                                                "question": sub_question})
        rag_results.append(answer)

    return rag_results, sub_questions

answers, questions = individually_decomposition(question, rag_prompt)

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


context = format_qa_pairs(questions, answers)

# Prompt
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = PromptRunnable(template)

@traceable
def run_individually(question):
    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"context":context,"question":question})

res = run_individually(question)
print(res)
