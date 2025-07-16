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
当前在线
姓名:杜贺飞  手机认证
岁
求职状态:
求职中
期望职位:
网络运营专员/助理、测试员、技术专员/助理、文员、Java开发工程师
期望地点:
北京
期望薪资:    -
举报 若该简历是无效简历,你可以在此举报
近两周
活跃度 高
更新简历
主动申请0个职位
查看过0家企业电话
关注度 高
被浏览0次  收到面试邀请0次
被下载0次  对TA感兴趣0个
教育经历
大连东软信息学院 |  2019年07月毕业
物联网工程
自我评价
本人工作踏实、严谨、专注,实际分析问题和处理问题的能力较强,工作的计划性、协调性及条理性较好,有旺盛的求知欲和进取心,善于独立思考,有良好的团队意识.在生活中,我尊重他人,能够和别人友好相处,我能够很快的学习新知识,有充足的信心胜任该工作.
"""]

#print(text[3])
result =  async_inference(resumes)
end=time.time()
print(result)
print(end - start)


