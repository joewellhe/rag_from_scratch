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
# res = query_analyzer.invoke({"question": "videos on chat langchain published in 2023 and exceed 10000 times viewed"})
# print(res)

sql_message = '''
你是一个擅长将用户描述的问题转换为database查询语句的专家，这次你需要处理的数据是
mysql, 数据库表的创建语句如下

CREATE TABLE `smbms_provider` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `proCode` varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '供应商编码',
  `proName` varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '供应商名称',
  `proDesc` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '供应商详细描述',
  `proContact` varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '供应商联系人',
  `proPhone` varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '联系电话',
  `proAddress` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '地址',
  `proFax` varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL COMMENT '传真',
  `createdBy` bigint(20) DEFAULT NULL COMMENT '创建者（userId）',
  `creationDate` datetime DEFAULT NULL COMMENT '创建时间',
  `modifyDate` datetime DEFAULT NULL COMMENT '更新时间',
  `modifyBy` bigint(20) DEFAULT NULL COMMENT '更新者（userId）',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

这里是几个样例数据
INSERT INTO `smbms_provider` VALUES ('1', 'BJ_GYS001', '北京三木堂商贸有限公司', '长期合作伙伴，主营产品:茅台、五粮液、郎酒、酒鬼酒、泸州老窖、赖茅酒、法国红酒等', '张国强', '13566667777', '北京市丰台区育芳园北路', '010-58858787', '1', '2013-03-21 16:52:07', null, null);
INSERT INTO `smbms_provider` VALUES ('2', 'HB_GYS001', '石家庄帅益食品贸易有限公司', '长期合作伙伴，主营产品:饮料、水饮料、植物蛋白饮料、休闲食品、果汁饮料、功能饮料等', '王军', '13309094212', '河北省石家庄新华区', '0311-67738876', '1', '2016-04-13 04:20:40', null, null);
INSERT INTO `smbms_provider` VALUES ('3', 'GZ_GYS001', '深圳市泰香米业有限公司', '初次合作伙伴，主营产品：良记金轮米,龙轮香米等', '郑程瀚', '13402013312', '广东省深圳市福田区深南大道6006华丰大厦', '0755-67776212', '1', '2014-03-21 16:56:07', null, null);
I
给定一个用户问题，你要将它转换为查询该表的sql语句，其中你需要选择合适的字段作为条件
如果不存在涉及到该表的信息，你需要返回 null

问题是{question}
'''
prompt = PromptRunnable(sql_message)
sql_analyzer = prompt | llm
res = sql_analyzer.invoke({"question": "生产食品的公司有哪些？"})
print(res)


