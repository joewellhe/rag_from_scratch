# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 14:32
# @Author  : HeJwei
# @FileName: logical_route.py
# Data model
from my_util.util import llm, PromptRunnable
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# Prompt
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source.

please output a answer which is one of ["python_docs", "js_docs", "golang_docs"], if there is no any correct

answer in this list, please output "other"

questions is {question}
"""

# LLM with function call
prompt = PromptRunnable(system)

@traceable
def run(q):
    router = prompt | llm | StrOutputParser()

    def choose_route(result):
        if "python_docs" == result:
            ### Logic here
            return "chain for python_docs"
        elif "js_docs" == result:
            ### Logic here
            return "chain for js_docs"
        else:
            ### Logic here
            return "golang_docs"

    from langchain_core.runnables import RunnableLambda

    full_chain = router | RunnableLambda(choose_route)

    return full_chain.invoke({"question": question})

question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

print(run({"question":question}))


