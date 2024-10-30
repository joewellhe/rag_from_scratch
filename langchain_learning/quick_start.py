from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain import LLMChain, ConversationChain
import os


os.environ["QIANFAN_ACCESS_KEY"] = "8558196f76744b0899d33ecc4745f2f4"
os.environ["QIANFAN_SECRET_KEY"] = "dfef8d05268042739c0fb03dadd9a6a5"

# 百度模型
llm = QianfanLLMEndpoint(model="ERNIE-3.5-8K", streaming=True, temperature=0.1)
# res = llm.invoke("用50个字左右阐述，生命的意义在于")
# print(res)

prompt = PromptTemplate(
    input_variables=["food"],
    template="What are 5 vacation destinations for someone who likes to eat {food}?"
)

# print(prompt.format(food="fruit"))
# chain = LLMChain(llm=llm, prompt=prompt)
#
# print(chain.run("fruit"))

# Memory: Add state to chains and agents
conversation = ConversationChain(llm=llm, verbose=True)
res= conversation.predict(input="hi")
print(res)
res = conversation.predict(input="I'm doing well! Just having a conversation with an AI")
print(res)
res = conversation.predict(input="What was the first thing I said to you?")
print(res)
res = conversation.predict(input="what is an alternative phrase for the first thing I said to you?")
print(res)
llm.chat