# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Any
from openai import OpenAI
from PROMPT import *

client = OpenAI(api_key="sk-c66c2a62e07b4312addb08325d3c24ab", base_url="https://api.deepseek.com")
def qa_openai(prompt: str) -> str:

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0
    )
    return response.choices[0].message.content


def create_multi_aspect_query(local_summary: str, local_detail: str) -> List[str]:
    base = (local_summary or "").strip()
    detail = (local_detail or "").strip()
    json_format = """{概述1:具体处置策略1};{概述2:具体处置策略2};....."""
    prompt = MULTI_ASPECT_QUERY.format(
        base=base,
        detail=detail,
        json_format=json_format
    )
    while True:
        try:
            raw_query = qa_openai(prompt)
            if ";" in raw_query:
                break
        except TypeError:
            print("Error")

    return raw_query

def to_list(raw_query: str) -> List[dict]:
    # 先去掉首尾空格，再按分号分隔
    segments = raw_query.strip().split(";")

    result = []
    for seg in segments:
        seg = seg.strip().strip("{}")  # 去掉两边空格和花括号
        if ":" in seg:
            k, v = seg.split(":", 1)  # 只分割一次，避免 value 中有冒号的问题
            result.append({k.strip(): v.strip()})

    return result



if __name__ == "__main__":
    items = [
        {"summary": "暴雨洪涝应急响应提升", "detail": "根据降雨量阈值将应急响应由Ⅲ级提升至Ⅱ级，启动转移安置与堤防巡查。"},
        {"summary": "地震灾害现场指挥", "detail": "成立现场前方指挥部，统一调度力量，开展搜索救援与医疗救治。"},
        {"summary": "城市内涝处置", "detail": "对易涝点位实行交通管制，排水部门启用应急泵站，实时发布出行提示。"},
        {"summary": "水库防汛调度", "detail": "对上游水库实施预泄洪调度，滚动研判下泄流量，做好下游预警。"},
        {"summary": "森林火灾扑救", "detail": "封控火线两侧，开设隔离带，组织航空灭火力量；同时开展群众避险转移。"},
    ]

    local_summary = "突发洪水官方通报"
    local_detail = "根据最新雨情水情，提升防汛应急响应等级并组织人员转移安置。"

    raw_queries = create_multi_aspect_query(local_summary, local_detail)
    print(raw_queries)
    queries = to_list(raw_queries)
    for query in queries:
        print(query)
