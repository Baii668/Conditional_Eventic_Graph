# -*- coding: utf-8 -*-

import openai
from openai import OpenAI
from typing import List, Dict, Optional, Any
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
import ast
from PROMPT import NORMAL_PROMPT, LOW_SELF_CHECK

"""
    输入是刚才all_data中的文件
    输出也写入其中原文件
    新增内容:
    "new_basis":自动生成的决策依据
"""




OPENAI_API_KEY = "sk-XXX"
# 很多代理需要带 /v1；如果你的网关不需要，请改回不带 /v1 的根地址
OPENAI_BASE_URL = "https://api.vveai.com/v1"

# 根据网关可用模型填写；如果网关不支持 gpt-4-turbo-2024-04-09，请换成它支持的名字
OPENAI_MODEL_NAME = "gpt-4-turbo-2024-04-09"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def get_openai_completion(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL_NAME,
    max_tokens: Optional[int] = 3000,
    temperature: Optional[float] = 0,
    stop: Optional[Any] = None,
    seed: Optional[int] = 123,
    tools: Optional[list] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
) -> str:
    """
    返回字符串内容而不是原始对象；同时过滤 None 参数，避免某些网关 400。
    """
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    # 过滤掉值为 None 的键
    clean_params = {k: v for k, v in params.items() if v is not None}

    try:
        completion = client.chat.completions.create(**clean_params)
        return completion.choices[0].message.content or ""
    except Exception as e:
        # 把异常信息抛出去或按需处理
        raise RuntimeError(f"OpenAI chat completion failed: {e}") from e


def qa_openai(prompt: str, model_name: str = OPENAI_MODEL_NAME, temperature: float = 0) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful planner."},
        {"role": "user", "content": prompt},
    ]
    # 统一走上面的封装，避免两套代码路径不一致
    return get_openai_completion(messages, model=model_name, temperature=temperature)

def qa_message_openai(prompt: str, prompt_2: str, model_name: str = OPENAI_MODEL_NAME, temperature=0):
    messages = [
        {"role": "system", "content": "You are a helpful planner."},
        {"role": "user", "content": prompt},
        {"role": "system", "content": prompt_2}
    ]
    response = get_openai_completion(messages=messages, model=model_name, temperature=temperature).choices[0].message.content
    return response

def low_self_check(event_summary, old_answer):
    new_thought = ""
    pattern = r"我观察到(.*?)。"
    matches = re.findall(pattern, str(old_answer))
    print(matches, str(len(matches)))
    i = 0
    yes_or_no = "是"
    if len(matches) != 0:
        while i < len(matches) and yes_or_no == "是":
            thought_1 = "我观察到" + matches[i] + "。"
            print("thought_1:\t", thought_1)
            low_self_check_prompt = LOW_SELF_CHECK.format(
                event_summary=event_summary,
                thought_1=thought_1
            )
            while True:
                try:
                    answer = qa_openai(low_self_check_prompt)
                    match_1 = re.search(r'\[是否符合定义\]:(.*?)\[改正后的分析\]:', answer, re.DOTALL)
                    if match_1:
                        yes_or_no = match_1.group(1).strip()
                        print(yes_or_no)
                        if yes_or_no != "是":
                            match2 = re.search(r'\[改正后的分析\]:(.*)', answer, re.DOTALL)
                            if match2:
                                new_thought += match2.group(1) + "\n"
                                break
                        else:
                            new_thought += thought_1 + "\n"
                            break
                    else:
                        new_thought += thought_1 + "\n"
                        break
                except TypeError:
                    print("TypeError ")
                # except openai.BadRequestError:
                #     print("openai.BadRequestError")
            i += 1

    # print("ok,return")
    return new_thought

def generate_basis(prompt, event_summary):
    answer = qa_openai(prompt)
    print("answer:\t", answer)

    thoughts = ""
    new_thought = low_self_check(event_summary, answer)
    print("new_thought:\t", new_thought)

    while new_thought.strip():
        thoughts += new_thought + "\n"
        print("thoughts:\t", thoughts)

        low_thought_answer = qa_message_openai(prompt, thoughts)
        new_thought = low_self_check(event_summary, low_thought_answer)
        print("new_thought:\t", new_thought)
        print("low_thought_answer:\t", low_thought_answer)

    print("low_thought_answer:\t", low_thought_answer)
    print("thought:\t", thoughts)

    while True:
        try:
            print("hi")
            match3 = re.search(r'(\{.*?\})', low_thought_answer.strip(), re.DOTALL)
            output = match3.group(1).strip()
            print(output)
            output_dict = ast.literal_eval(output)
            data = {
                "event": event_summary,
                "thought": thoughts,
                "basis": output_dict
            }
            break
        except AttributeError:
            low_thought_answer = qa_openai(prompt)
            print("AttributeError")
        except SyntaxError:
            low_thought_answer = qa_openai(prompt)
            print("SyntaxError")

    return data


if __name__ == '__main__':
    data_path = "./all_data/"

    # names = []
    # for name in os.listdir(output_path):
    #     names.append(name)
    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            file_path = os.path.join(data_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "new_basis" not in data:
                print("Processing:###", filename)
                event_summary = data.get("summary", "")
                if len(event_summary) != 0:

                    prompt = data.get("prompt", "")

                    event_thought_basis = generate_basis(prompt, event_summary)

                    data["new_basis"] = event_thought_basis["basis"]

                    with open(file_path, "w", encoding="utf-8") as ft:
                        json.dump(data, ft, ensure_ascii=False)




