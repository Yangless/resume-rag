import time

import pandas as pd
import uvicorn
from fastapi import FastAPI
from concurrent.futures import ProcessPoolExecutor
import asyncio
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class Config:
    model_path = "qwen2.5-1.5B"
    output_dir = "qwen2.5-resume-finetune1"
config = Config()

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "qwen2.5-1.5B-merge1",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.output_dir,
        trust_remote_code=True
    )
    return model,tokenizer

MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class TextRequest(BaseModel):
    text:list
app = FastAPI()

def async_inference(input_texts):
    loop = asyncio.get_event_loop()
    result = run_inference_on_gpu(input_texts)
    return result


def extract_first_json(text):
    import re
    pattern = r'\{.*?\}'
    match = re.findall(pattern, text, flags=re.DOTALL)[1]
    return match

import threading

local_data = threading.local()



def run_inference_on_gpu(resumes):
    try:
        if not hasattr(local_data, 'model') or local_data.model is None:
            local_data.model,local_data.tokenizer= load_model()
        input_texts = []
        for resume in resumes:
            input_text = f"""从下面这段简历文本中提取涉及以下7类标签的总结：基本信息,教育经历,工作经历,项目经历,培训经历,个人评价,技能证书。
            每份工作或项目经历字数不超过100字;如果标签没有提取到内容则返回空字符串;不要嵌套JSON。
    简历文本:
    {resume}

    返回JSON格式:
    ```json
    {{"基本信息":"","教育经历":[],"工作经历":[],"项目经历":[],"培训经历":[],"个人评价":"","技能证书":""}}
    ```
    """
            input_texts.append(input_text)
            # print(input_text)

        print(f"正在批量处理 {len(input_texts)} 条简历...")
        # Tokenize 批量输入
        inputs = local_data.tokenizer(
            input_texts,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        ).to('cuda')

        with torch.no_grad():
            local_data.model.generation_config.do_sample = False  # 启用采样
            local_data.model.generation_config.temperature = None  # 适度随机性
            local_data.model.generation_config.top_p = None  # 核采样(nucleus sampling)
            local_data.model.generation_config.top_k = None  # 限制候选词数量

            outputs = local_data.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=local_data.tokenizer.eos_token_id,
                eos_token_id=local_data.tokenizer.eos_token_id,
            )

        responses = local_data.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(responses)
        cleaned_responses = [extract_first_json(r) for r in responses]
        return cleaned_responses
    except Exception as e:
        print(e)


start=time.time()
chunk_size=20
for chunk in pd.read_csv("./db_ai.t_ck_cand_con0204_trunk.csv/db_ai.t_ck_cand_con0204_trunk.csv",
                             chunksize=chunk_size):
    resumes=list(chunk.iloc[-2:,2])
    break
resumes=["""
个人简历 简历信息：孙双永 8-9年工作经验 | 男 | 33岁(1983年7月 15日) | 已婚 | 170cm (ID:310051942) 居住地: 北京-昌平区 户 口: 郑州 地 址: 昌平区西二旗 (邮编:000000) 电 话: 13524392655(手机) E-mail: sun007700@126.com 最近工作 [1年3个月]
"""]

#print(text[3])
result =  async_inference(resumes)
end=time.time()
print(result)
print(end - start)


