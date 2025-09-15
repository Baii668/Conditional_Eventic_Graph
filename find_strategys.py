import openai
from openai import OpenAI
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
import ast
from itertools import combinations
import random

OPENAI_API_KEY = "XXX"
OPENAI_BASE_URL = "https://api.v3.cm/v1"
OPENAI_MODEL_NAME = "gpt-4"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

from PROMPT import FIND_STRATEGYS, example
from generate_condition import qa_openai


"""
    这个就是简单写了个prompt强调以“高校”为决策主体做出了哪些决策
    输入输出都是在all_data这个文件夹下，新增内容为"new_outcome"
"""




def iterate_basis(basis):
    for key, values in basis.items():
        yield key, values

def extract_chinese_characters(text):
    # 使用正则表达式匹配汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', text)
    return ''.join(chinese_characters)





def find_strategys(event_summary, event, example):

    prompt = FIND_STRATEGYS.format(
        event_summary=event_summary,
        event=event,
        example=str(example)
    )
    while True:
        try:
            answer = qa_openai(prompt)
            print(answer)

            answer = answer.replace("'", '"')
            match = re.search(r'(\{[\s\S]*?\})', answer, re.DOTALL)
            if match:
                out_answer = match.group(1).strip()
                print(out_answer)
                dict_answer = json.loads(out_answer)
                break
            else:
                dict_answer = {}
                break
        except JSONDecodeError:
            print("JSONDecodeError")

    return dict_answer


if __name__ == '__main__':
    folder_path = "./all_data/"
    example = example
    # names = []
    # for name in os.listdir(output_path):
    #     names.append(name)

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            event_summary = data["summary"]
            events = data["cases"]

            event_str = ""
            for event in events:
                event_str += event["relevant_context"]

            if "new_outcome" not in data:

                new_outcome = find_strategys(event_summary, event_str, example)
                print(str(new_outcome))

                data["new_outcome"] = new_outcome

                with open(file_path, "w", encoding="utf-8") as fs:
                    json.dump(data, fs, ensure_ascii=False)
                    print("###写入", filename, "###")
