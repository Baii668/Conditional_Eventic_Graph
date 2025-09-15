import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig, AutoModel
import pandas as pd
from evaluate import load
from PROMPT import ICL_PROMPT, NORMAL_PROMPT
from scipy.spatial.distance import cosine
import openai
from openai import OpenAI

from generate_condition import generate_basis



'''
    输入是NEWcasev3_1中的数据
    记得创建all_data文件夹
    输出是all_data文件夹下中和NEWcasev3_1同名的json
    
    相比原始数据，新增的内容为：
    "prompt":结合了找到的上下文示例
'''



sim_tokenizer = AutoTokenizer.from_pretrained("/home/jianbaizhao/model/simbert-base-chinese")
sim_model = AutoModel.from_pretrained("/home/jianbaizhao/model/simbert-base-chinese")

OPENAI_API_KEY = "XXXX"
OPENAI_BASE_URL = "https://api.v3.cm/v1"
OPENAI_MODEL_NAME = "gpt-4o"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def find_top_similar(input_text, data, top_n=10):
    """
    从 data 中挑选出与 input_text 语义最相似的 top_n 条内容。
    :param input_text: 输入文本，用于比较的基准
    :param data: 数据集，每个元素是一个文本（或字典）
    :param top_n: 需要返回的最相似条目的数量
    :return: 最相似的 top_n 条内容
    """
    # 计算每个 data 项与 input_text 的相似度
    similarity_scores = []
    for item in data:
        text = item["event"]  # 比较基于 "event" 内容
        sim_score = compare(input_text, text)
        similarity_scores.append((item, sim_score))

    # 根据相似度排序，取前 top_n 条
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_items = [item[0] for item in similarity_scores[:top_n]]
    return top_items


def compare(a, b):
    texts = [
        a,
        b
    ]
    inputs = sim_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = sim_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    cosine_sim = 1 - cosine(embeddings[0], embeddings[1])

    return cosine_sim

def calculate_perplexity(model, tokenizer, text):
    # 将文本转为输入张量
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to("cuda")

    # 获取模型输出，包含损失值
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Cross-entropy loss
        loss = outputs.loss

    # 计算困惑度
    perplexity = torch.exp(loss)
    return perplexity.item()



def generate_combinations(data, combination_size):
    result = []
    temp = []

    def backtrack(start):
        if len(temp) == combination_size:
            result.append(temp[:])  # 添加当前组合的副本
            return
        for i in range(start, len(data)):
            temp.append(data[i])
            backtrack(i + 1)  # 递归选择下一个元素
            temp.pop()  # 回溯，撤销选择

    backtrack(0)
    return result

# 处理输入文本
def process_input_texts(input_text, data, model, tokenizer):
    # 生成所有 5 项组合
    # combinations = generate_combinations(data[:10], 3)  # 假设这里只取了前 10 项数据
    # print("Total combinations count:", len(combinations))

    # 存储最低分数和相应的组合
    # min_perplexity = float('inf')
    # best_combo = None

    print(f"Processing input text: {input_text}")

    top_similar_data = find_top_similar(input_text, data, top_n=10)
    print(f"Top similar data found: {top_similar_data}")
    combinations = generate_combinations(top_similar_data, combination_size=3)  # 组合大小为 3
    print("Total combinations count:", len(combinations))

    min_perplexity = float('inf')
    best_combo = None

    normat_prompt = NORMAL_PROMPT.format(
        event_summary=input_text
    )

    # 遍历每个组合，计算其 perplexity
    for combo in combinations:
        combo_text = ""
        for item in combo:
            # 组合的内容
            a_example = ICL_PROMPT.format(
                event_summary=item["event"],
                thoughts=item["thought"],
                basis=item["basis"]
            )
            combo_text += a_example + "\n"

        combo_text += normat_prompt

        # 计算该组合的 perplexity 分数
        a_score = calculate_perplexity(model, tokenizer, combo_text)
        # print(str(a_score))
        # 如果当前组合的 perplexity 分数低于最小值，更新最小值和最佳组合
        if a_score < min_perplexity:
            min_perplexity = a_score
            best_combo = combo

    # 输出最低 perplexity 和相应的组合
    print(f"Best combination with lowest perplexity: {best_combo}")
    print(f"Lowest perplexity score: {min_perplexity}")
    return best_combo


def get_prompt(event_summary, combined_data, model, tokenizer):
    best_combo = process_input_texts(event_summary, combined_data, model, tokenizer)

    output = ""
    for item in best_combo:
        # 组合的内容
        best_example = ICL_PROMPT.format(
            event_summary=item["event"],
            thoughts=item["thought"],
            basis=item["basis"]
        )
        output += best_example + "\n"
    normat_prompt = NORMAL_PROMPT.format(
        event_summary=event_summary
    )
    output += normat_prompt

    return output



if __name__ == '__main__':

    model_name = "/home/share/models/Qwen2.5-14B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    perplexity = load("/home/jianbaizhao/CodingFile/metrics/perplexity/perplexity.py", module_type="metric")


    with open("combined_data.json", "r", encoding="utf-8") as f:
        combined_data = json.load(f)

    data_path = "/home/jianbaizhao/CodingFile/IDM/data/assemble_dto_data_v2_tag"
    output_path = "./all_data/"
    names = []
    for name in os.listdir(output_path):
        names.append(name)
    for filename in os.listdir(data_path):
        if filename.endswith(".json") and filename not in names:
            file_path = os.path.join(data_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            event_summary = data.get("summary", "")

            prompt = get_prompt(event_summary, combined_data, model, tokenizer)

            data["prompt"] = prompt

            with open(output_path + filename, "w", encoding="utf-8") as ft:
                json.dump(data, ft, ensure_ascii=False)






