# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
import ast
from PROMPT import NORMAL_PROMPT, LOW_SELF_CHECK
from itertools import combinations
import random


# OPENAI_API_KEY = "sk-6uuyLGOBjYS6VC60551eF90f89A34b45A5E4B592A936Ea5a"
# OPENAI_BASE_URL = "https://api.vveai.com/v1"
# OPENAI_MODEL_NAME = "gpt-4-turbo-2024-04-09"
# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

from PROMPT import FIND_FACTORS, FIND_RELATION, Causal_Intervention_1, Causal_Intervention_2
# from generate_condition import qa_openai
#
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



"""
    输入"all_data"文件夹中的json
    记得新建out_data文件夹
    输出写在out_data中与all_data中内容同名的文件
    新增内容:
    "new_edges":以start（一些决策依据）和end（一个官方策略）形式存储的边，表示他们之间的关系
"""




def iterate_basis(basis):
    for key, values in basis.items():
        yield key, values

def extract_chinese_characters(text):
    # 使用正则表达式匹配汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', text)
    return ''.join(chinese_characters)


def generate_subsets(input_list):
    """
    生成所有含有两个以上元素的子集组合（不使用 itertools），并按长度从小到大排序。
    :param input_list: 输入列表
    :return: 所有含有两个以上元素的子集组合，按长度从小到大排序
    """
    result = []

    def backtrack(start, subset):
        # 如果当前子集长度大于等于 2，就添加到结果
        if len(subset) >= 2:
            result.append(subset[:])

        # 从 start 开始遍历 input_list
        for i in range(start, len(input_list)):
            # 选择当前元素

            subset.append(input_list[i])

            # 递归构建剩余子集
            backtrack(i + 1, subset)

            # 移除最后一个元素，回溯
            subset.pop()

    # 从空集开始构建
    backtrack(0, [])

    # 按子集长度从小到大排序
    result.sort(key=len)

    return result




def find_factors(event_summary, strategys, basis):

    edges = []
    for strategy in strategys:
        a_start = []
        for key, value in iterate_basis(basis):
            if value != "未找到相关信息":
                prompt = FIND_RELATION.format(
                    event=event_summary,
                    dependence=key,
                    factor=value,
                    strategy=strategy["outcome_tag"] + ": " + strategy["summary"]
                )
                while True:
                    try:
                        print(prompt)
                        answer_2 = qa_openai(prompt)
                        answer_2 = extract_chinese_characters(answer_2)
                        print("answer2:\t", answer_2)
                        if answer_2.strip() in ["相关", "不相关"]:
                            break
                    except Exception:
                        print("Reload!!!!")

                if answer_2.strip() == "相关":
                    a_start.append(key)
                    print(strategy["outcome_tag"] + ": " + strategy["summary"] + "和" + key + "是" + value + answer_2)
        edges.append({
            "start": a_start,
            "end": strategy
        })
    print("edges:\t")
    print(str(edges))
    new_deges = []
    for edge in edges:

        all_starts = generate_subsets(edge["start"])
        final_start = []

        check_yes_or_no = "否"
        # random.shuffle(all_starts)

        for all_start in all_starts:

            print(str(all_start))
            print(edge["end"])
            delete_basis_str = ""
            for a_basis in edge["start"]:
                if a_basis not in all_start:
                    delete_basis_str += a_basis + "是" + basis[a_basis] + ","
            all_basis_str = ""
            for b in edge["start"]:
                all_basis_str += b + "是" + basis[b] + ","
            a_basis_str = ""
            for a_basis in all_start:
                a_basis_str += a_basis + "是" + basis[a_basis] + ","

            while True:
                try:
                    prompt_1 = Causal_Intervention_1.format(
                        event_summary=event_summary,
                        strategy=edge["end"],
                        basis=all_basis_str,
                        delete_basis_str=delete_basis_str
                    )
                    answer_3 = qa_openai(prompt_1)
                    prompt_2 = Causal_Intervention_2.format(
                        event_summary=event_summary,
                        strategy=edge["end"],
                        basis=all_basis_str,
                        a_basis_str=a_basis_str
                    )
                    answer_4 = qa_openai(prompt_2)
                    print(answer_3)
                    print(answer_4)
                    match_2 = re.search(r'\[答案\][：:]\s*(.*)', answer_3, re.DOTALL)
                    match_3 = re.search(r'\[答案\][：:]\s*(.*)', answer_4, re.DOTALL)

                    if match_2 and match_3:
                        yes_or_no = match_2.group(1).strip()
                        yes_or_no_str = extract_chinese_characters(yes_or_no)
                        print("match_2:\t", yes_or_no_str)

                        another_yes_or_no = match_3.group(1).strip()
                        another_yes_or_no_str = extract_chinese_characters(another_yes_or_no)
                        print("match_3:\t", another_yes_or_no_str)
                        """
                            如果只考虑a_basis，是否足以支撑官方做出XX决策
                            所以，如果是“支持”的话，就说明当前组合是最小集合
                        """
                        if yes_or_no_str == "是" and another_yes_or_no_str == "是":
                            check_yes_or_no = "是"
                            final_start = all_start

                        break
                    else:
                        continue
                except TypeError:
                    print("reload")


            if check_yes_or_no == "是":
                new_deges.append({
                    "new_start": final_start,
                    "new_end": edge["end"]
                })
                break
            else:
                continue
        if check_yes_or_no != "是":
            new_deges.append({
                "new_start": edge["start"],
                "new_end": edge["end"]
            })



    print("NEW_edges:\t")
    print(str(new_deges))
    return new_deges


if __name__ == '__main__':
    folder_path = "./all_data/"
    out_path = "./new_out_data_0902/"
    names = []
    for name in os.listdir(out_path):
        names.append(name)

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename not in names:
            print(filename)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            strategys = data["outcome_new"]
            basis = data["new_basis"]
            event_summary = data["summary"]

            # a_new_start = data["new_edges"][0]["new_start"]
            # if len(a_new_start) != 0:
            edges = find_factors(event_summary, strategys, basis)

            data["new_edges"] = edges

            with open(out_path+filename, "w", encoding="utf-8") as fs:
                json.dump(data, fs, ensure_ascii=False)
                print("###写入", filename, "###")
