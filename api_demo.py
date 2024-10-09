# -*- coding: utf-8 -*-
# @Time    : 2024/9/17 14:21
# @Author  : HeJwei
# @FileName: api_demo.py

print("hello world")

import os
import qianfan

# 通过环境变量初始化认证信息
# 方式一：【推荐】使用安全认证AK/SK鉴权
# 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk，如何获取请查看https://cloud.baidu.com/doc/Reference/s/9jwvz2egb
os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"

# 方式二：【不推荐】使用应用AK/SK鉴权
# 替换下列示例中参数，将应用API_Key、应用Secret key值替换为真实值
#os.environ["QIANFAN_AK"] = "应用API_Key"
#os.environ["QIANFAN_SK"] = "应用Secret_Key"

# chat_comp = qianfan.ChatCompletion()
#
# # 指定特定模型
# resp = chat_comp.do(model="ERNIE-3.5-8K", messages=[{
#     "role": "user",
#     "content": "你好"
# }])
#
# print(resp["body"])

# Embedding
emb = qianfan.Embedding()
resp = emb.do(model="Embedding-V1", texts=[
    "推荐一些美食","给我讲个故事"
])
print(resp["body"])