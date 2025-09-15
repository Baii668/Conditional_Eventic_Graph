# -*- coding: utf-8 -*-
"""
之前的步骤是针对每个事件生成一个具体的图谱--对决策生成的解释
此处的终极目的是：对同类事件的相同决策做归纳，形成抽象规律，即多数情况下做出这个决策需要的前置条件是什么（通常情况），另外少数情况下还需要考虑XXX
按理说大约的步骤是：
Step1：根据相同决策把依据的条件都检索出来
Step2：根据条件的aspect分类，比如这一类都是考虑事件地点和事件后果的，那一类都是考虑责任主体的
Step3：假设都是考虑事件地点，现在有多个具体的条件的value，就应该去概括（归类），此时可以通过多个角度描述，借鉴之前写的反事实的操作
    Step3.1：每组条件-决策对都生成m个概括角度
    Step3.2：从中挑选并集？或者能全部概括所有的角度集合m-
    Step3.3：生成反事实数据，挑选其中n个重要的角度（相当于对前两步做精炼）on
Step4：根据数量上的统计条件是通用的or特殊的
"""

from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel  # 如需重排，可再引入 FlagReranker
import numpy as np
import math
from typing import List, Dict, Tuple, Literal, Optional, Any
from PROMPT import *
import re

client = OpenAI(api_key="sk-XXX", base_url="https://api.deepseek.com")
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

    def to_list(raw_query: str) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        if not raw_query:
            return result

        # 按中英文分号切分
        segments = re.split(r"[;；]+", raw_query.strip())

        for seg in segments:
            seg = seg.strip().strip("{}")
            if not seg:
                continue

            # 统一成英文冒号
            if ":" not in seg and "：" in seg:
                seg = seg.replace("：", ":", 1)

            if ":" in seg:
                k, v = seg.split(":", 1)
                k, v = k.strip(), v.strip()
                if k or v:
                    result.append({k: v})

        return result
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

def to_list(raw_query: str) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    if not raw_query:
        return result

    # 按中英文分号切分
    segments = re.split(r"[;；]+", raw_query.strip())

    for seg in segments:
        seg = seg.strip().strip("{}")
        if not seg:
            continue

        # 统一成英文冒号
        if ":" not in seg and "：" in seg:
            seg = seg.replace("：", ":", 1)

        if ":" in seg:
            k, v = seg.split(":", 1)
            k, v = k.strip(), v.strip()
            if k or v:
                result.append({k: v})

    return result


MatchMode = Literal["any", "all"]

class BGERecallOnly:
    """
    用 bge-m3 做“只召回不排序”的极简索引与检索：
    - build_index(List[str])
    - recall(queries: List[str], thresholds, use_dense/sparse, match_mode)
    只根据阈值过滤是否命中，按语料原始顺序返回结果，不做排序。
    """

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _sparse_dot(q: Dict[int, float], d: Dict[int, float]) -> float:
        # 词面权重点积（BGEM3 的 lexical_weights）
        if not q or not d:
            return 0.0
        if len(q) > len(d):
            q, d = d, q
        s = 0.0
        for k, v in q.items():
            if k in d:
                s += v * d[k]
        return float(s)

    def __init__(self, model_dir: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_dir, use_fp16=use_fp16)
        self.texts: List[str] = []
        self._dense: List[np.ndarray] = []
        self._sparse: List[Dict[int, float]] = []
        self.metas: List[Dict[str, Any]] = []

    # ---------------------------
    # 1) 建索引
    # ---------------------------
    def build_index(self, texts: List[str], metas: Optional[List[Dict[str, Any]]] = None):
        self.texts = [(t or "").strip() for t in texts]
        enc = self.model.encode_corpus(
            self.texts,
            return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        self._dense  = [np.asarray(v, dtype=np.float32) for v in enc["dense_vecs"]]
        self._sparse = enc["lexical_weights"]
        self.metas   = [{} for _ in self.texts] if metas is None else metas



if __name__ == "__main__":
    items = [
        {"summary": "暴雨洪涝应急响应提升", "detail": "根据降雨量阈值将应急响应由Ⅲ级提升至Ⅱ级，启动转移安置与堤防巡查。"},
        {"summary": "地震灾害现场指挥", "detail": "成立现场前方指挥部，统一调度力量，开展搜索救援与医疗救治。"},
        {"summary": "城市内涝处置", "detail": "对易涝点位实行交通管制，排水部门启用应急泵站，实时发布出行提示。"},
        {"summary": "水库防汛调度", "detail": "对上游水库实施预泄洪调度，滚动研判下泄流量，做好下游预警。"},
        {"summary": "森林火灾扑救", "detail": "封控火线两侧，开设隔离带，组织航空灭火力量；同时开展群众避险转移。"},
    ]

    local_summary = "突发洪水官方通报"
    local_detail  = "根据最新雨情水情，提升防汛应急响应等级并组织人员转移安置。"
    model_dir     = "D:\\models\\BAAI\\bge-m3"
