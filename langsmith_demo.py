# -*- coding: utf-8 -*-
# @Time    : 2024/9/18 21:05
# @Author  : HeJwei
# @FileName: langsmith_demo.py

from langsmith.wrappers import wrap_openai
from langsmith import traceable
import qianfan
import os

os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Auto-trace LLM calls in-context
chat_comp = qianfan.ChatCompletion()


@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = chat_comp.do(
        model="ERNIE-3.5-8K",
        messages=[{
            "role": "user",
            "content": "你好"
        }]
    )

    return result['result']


pipeline("Hello, world!")