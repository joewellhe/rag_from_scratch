# -*- coding: utf-8 -*-
# @Time    : 2024/10/6 19:12
# @Author  : HeJwei
# @FileName: step_back.py
from my_util.util import PromptRunnable
from my_util.util import llm
from my_util.util import QianfanOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langsmith import traceable
import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# Few Shot Examples 给定一些案例来构造prompt
few_shot_step_back_template = '''
你是一位世界知识专家。你的任务是后退一步，将一个问题重新表述为更具普遍性的问题，使其更容易回答。以下是一些例子：
样例1
输入: 警察成员可以执行合法的拘捕么？
输出: 警察成员可以执行的行政措施有哪些？

样例2
输入: 詹姆斯出生在哪个国家？
输出: 詹姆斯的个人履历是什么？

原始问题: {question}
'''
step_back_prompt = PromptRunnable(few_shot_step_back_template)
generate_queries_step_back = step_back_prompt | llm | StrOutputParser()
# question = "15岁的人故意杀人会受到什么样的处罚?"
# res = generate_queries_step_back.invoke({"question": question})

template = '''
你是一位世界知识专家。我将向你提出一个问题。你的回答应该全面且不与以下背景内容相矛盾（如果相关的话）。如果这些背景内容不相关，则可以忽略它们
背景内容是:
# {normal_context}
# {step_back_context}
问题是:
{question}
请你回答
'''
response_prompt = PromptRunnable(template)
vectorstore = Chroma(embedding_function=QianfanOpenAIEmbeddings(), persist_directory="../model/vectorstore")
retriever = vectorstore.as_retriever()

question = "16岁的人抢劫会受到什么样的处罚?"


@traceable()
def step_back(q):
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": q})

print(step_back(question))
print(llm(question))