import os
import asyncio
import pandas as pd
import tiktoken
import chainlit as cl
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
"""
"""


from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

# 全局变量
search_engine = None


@cl.on_chat_start
async def on_chat_start():
    global search_engine

    # 从环境变量中获取API密钥和模型名称
    api_key = os.environ.get("GRAPHRAG_API_KEY")
    if not api_key:
        await cl.Message(content="错误：GRAPHRAG_API_KEY 未设置").send()
        raise ValueError("GRAPHRAG_API_KEY 环境变量未设置")

    llm_model = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-3.5-turbo")

    # 初始化ChatOpenAI实例
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.AzureOpenAI,  # 使用OpenAI azure API
        api_base="https://lilith-plat-dify.openai.azure.com/",
        max_retries=20,  # 最大重试次数
        api_version="2024-05-01-preview",
    )

    # 初始化token编码器
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # 加载社区报告作为全局搜索的上下文
    INPUT_DIR = "artifacts"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    COMMUNITY_LEVEL = 2

    try:
        # 读取parquet文件
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    except FileNotFoundError as e:
        await cl.Message(content=f"错误：找不到所需的parquet文件。{str(e)}").send()
        return

    # 读取索引器报告和实体
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    await cl.Message(content=f"报告记录数: {len(report_df)}").send()

    # 基于社区报告构建全局上下文
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    # 设置上下文构建器参数
    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    # 设置map LLM参数
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    # 设置reduce LLM参数
    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    # 初始化全局搜索引擎
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    await cl.Message(content="全局搜索系统已准备就绪，请输入您的查询。").send()


@cl.on_message
async def main(message: cl.Message):
    global search_engine

    if search_engine is None:
        await cl.Message(content="搜索引擎尚未初始化。请稍后再试。").send()
        return

    query = message.content
    result = await search_engine.asearch(query)

    # 发送搜索结果
    await cl.Message(content=result.response).send()

    # 发送上下文数据报告
    context_data = f"上下文数据报告数量: {len(result.context_data['reports'])}"
    await cl.Message(content=context_data).send()

    # 发送LLM调用信息
    llm_info = f"LLM调用次数: {result.llm_calls}. LLM tokens数: {result.prompt_tokens}"
    await cl.Message(content=llm_info).send()


if __name__ == "__main__":
    cl.run()
