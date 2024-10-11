# -*- coding: utf-8 -*-
# @Time    : 2024/10/10 18:54
# @Author  : HeJwei
# @FileName: metadata_filter.py
from my_util.util import PromptRunnable, llm

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser

# docs = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
# ).load()
#
# print(docs[0].metadata)

system_message = """
You are an expert at converting user questions into database queries. You have access to a database of tutorial videos 
about a software library for building LLM-powered applications. 
Given a question, return a json string that follows the structure of the TutorialSearch model:

{{
    "content_search": str,  # Similarity search query applied to video transcripts.
    "title_search": str,    # Alternate version of the content search query to apply to video titles.
    "min_view_count": Optional[int],  # Minimum view count filter, inclusive.
    "max_view_count": Optional[int],  # Maximum view count filter, exclusive.
    "earliest_publish_date": Optional[str],  # Earliest publish date filter (YYYY-MM-DD), inclusive.
    "latest_publish_date": Optional[str],    # Latest publish date filter (YYYY-MM-DD), exclusive.
    "min_length_sec": Optional[int],  # Minimum video length in seconds, inclusive.
    "max_length_sec": Optional[int]   # Maximum video length in seconds, exclusive.
}}
Return only the json string as your output, no any one more character.

Here is a question:
{question}
"""
prompt = PromptRunnable(system_message)

query_analyzer = prompt | llm
res = query_analyzer.invoke({"question": "videos on chat langchain published in 2023 and exceed 10000 times viewed"})


sql_message = '''

'''