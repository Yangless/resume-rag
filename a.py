import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor,as_completed

import pandas as pd

from data_dict import resumes
import requests

logger = logging.getLogger("resume_api_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_file = "resume_api_logger.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def put_data(resume):
    return f""" 
<Instructions>
你是一个专业的人力资源（HR）信息提取AI助手。你的任务是仔细阅读一份简历文本，并从中提取关键信息，删除冗余内容，仅保留重要内容和有效要点，用简洁清晰的语言表达。，然后按照严格的JSON格式输出。

这是需要你分析的简历文本：
<resume_text>
{resume}
</resume_text>

请遵循以下规则进行信息提取和格式化：

1. **输出格式**: 你的最终输出**必须**是一个单一、完整、有效的JSON对象。不要在JSON对象前后添加任何说明性文字、注释或Markdown代码块（例如 json）。
2. **JSON结构**: 生成的JSON对象**必须**包含以下七个顶级键，且仅包含这七个键。请使用中文作为键名：
   - "基本信息": (对象) 包含姓名、电话、邮箱、求职意向等个人基本资料。
   - "教育经历": (对象数组 [{{}}]) 包含所有教育背景。每个对象应包含学校、专业、学历、在校时间等字段。
   - "工作经历": (对象数组 [{{}}]) 包含所有工作经验。每个对象应包含公司名称、职位、在职时间、工作内容等字段。
   - "项目经历": (对象数组 [{{}}]) 包含所有项目经验。每个对象应包含项目名称、项目角色、项目时间、项目描述等字段。
   - "培训经历": (对象数组 [{{}}]) 包含所有培训经历。每个对象应包含培训机构、课程名称、时间等。
   - "个人评价": (字符串 "") 包含简历中的自我评价或职业总结部分。
   - "技能证书": (对象) 包含专业技能和获得的证书。可以按技能类别分类，例如 {{"语言技能": [], "专业技能": [], "证书": []}}。
   - 所有内容必须经过**信息浓缩与语言简化**，避免冗长重复，直击重点。
   - 不要思考,不要调用工具
3. **处理缺失信息**: 如果简历中没有找到某个类别的信息（例如，没有任何“培训经历”），请不要省略该键。对于期望是对象数组的键（如教育经历, 工作经历等），请使用空数组 [] 作为其值；对于期望是字符串的键（如个人评价），请使用空字符串 "" 作为其值；对于期望是对象的键（如基本信息），请使用空对象 {{}}。</Instructions>
"""
resumes=resumes[0:1]
resumes_dict=[]
for t in range(10,61,10):
    resumes_dict.append(resumes*t)
content_len=len(resumes[0])

def access_token():
    GET_TOKEN_URL = 'https://aitest.wintalent.cn/chat/getAccessToken'
    pyload = {"appId": "test_max_propmpt",
              "securityKey": "WipDbZs8eHwWiawPM1nQiYSw1Rsf7RGW",
              "corpCode": "testcorp",
              "userId": "testcorp-10001"
              }
    response = requests.post(GET_TOKEN_URL, json=pyload)
    if response.status_code == 200:
        # print(response.json())
        accesstoken = response.json()['data']['token']
        return accesstoken


# API 配置

def ai_api(content,accesstoken_):
    CHAT_API_URL = "https://aitest.wintalent.cn/chat/completions/synchMsg"
    headers = {"wintalentaiToken": accesstoken_}
    try:
        prompt = content
        body = {"corpCode": "testcorp",
                "userId": "testcorp-10001",
                "sessionId": "testcorp-10001-20001",
                "bizType": 5101,
                "prompt": prompt
                }
        response = requests.post(CHAT_API_URL, json=body, headers=headers)
        data = response.json()['data']
        print(data)

    except Exception as e:
        print(f"处理出错: {e}")

def process_mul(chunk_, accesstoken_):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(ai_api, i, accesstoken_) for i in chunk_]




for resumes_dict_i in resumes_dict:
    accesstoken_ = access_token()
    start = time.time()
    process_mul(resumes_dict_i,accesstoken_)
    end=time.time()
    logger.info(f'{len(resumes_dict_i)}个文本平均长度:{content_len}')
    logger.info(f'耗时:{end-start}')

