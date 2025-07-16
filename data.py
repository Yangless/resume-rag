import json
import os
import re

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def ai_api(content, accesstoken_):
    CHAT_API_URL = "https://aitest.wintalent.cn/chat/completions/synchMsg"
    headers = {"wintalentaiToken": accesstoken_}
    content = content.replace('\\r', '\r').replace('：', ':').replace('\\n','\n')
    try:
        prompt = f"""从下面这段简历文本中提取涉及以下7类标签的总结：基本信息,教育经历,工作经历,项目经历,培训经历,个人评价,技能证书。
        每份工作或项目经历字数不超过100字;如果标签没有提取到内容则返回空字符串;不要嵌套JSON。
简历文本:
{content}

返回JSON格式:
```json
{{"基本信息":"","教育经历":[],"工作经历":[],"项目经历":[],"培训经历":[],"个人评价":"","技能证书":""}}
```
"""
        body = {"corpCode": "testcorp",
                "userId": "testcorp-10001",
                "sessionId": "testcorp-10001-20001",
                "bizType": 5101,
                "prompt": prompt
                }
        response = requests.post(CHAT_API_URL, json=body, headers=headers)
        data = response.json()['data']
        data_json=re.findall(r'json\n+(.*)```',data,re.DOTALL)[0]
        data_dict=json.loads(data_json)
        #print(data_dict)
        return pd.DataFrame({'resume_text':content,'summary':data_json},index=[0])

    except Exception as e:
        print(f"处理出错: {e}")

def process_mul(chunk_, accesstoken_):
    df_end = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(ai_api, chunk_.iloc[i, 2], accesstoken_) for i in range(len(chunk_))]
        for future in as_completed(futures):
            try:
                result = future.result()
                df_end = pd.concat([df_end, result], ignore_index=True, axis=0)
            except Exception as e:
                print(e, future)
    return df_end


if __name__ == '__main__':
    output_path = "./processed_output.csv"
    #if os.path.exists(output_path):
    #    os.remove(output_path)
    chunk_size = 5
    num = 0

    for chunk in pd.read_csv("./db_ai.t_ck_cand_con0204_trunk.csv/db_ai.t_ck_cand_con0204_trunk.csv",
                             chunksize=chunk_size):
        accesstoken = access_token()
        print(num)
        num += 1
        #if num>3:
            #break
        if num<80:
            continue

        chunk.iloc[:, 2] = chunk.iloc[:, 2].fillna('')
        processed_chunk = process_mul(chunk, accesstoken)
        processed_chunk.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path),
                               encoding='utf-8')

