import json
import logging
import re
import time

import argparse
import json5
import asyncio
import aiohttp
from data_dict import resumes
logger = logging.getLogger("resume_test_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_file = "resume_test_logger.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True, help='列表索引')
parser.add_argument('--end', type=int, required=False,default=80, help='range范围')
parser.add_argument('--full',action='store_false',required=False,default=True,help='全量样本')
args = parser.parse_args()

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
   - "基本信息": (对象) 包含姓名、求职意向等个人基本资料，删除所有联系方式（如电话、邮箱、地址、工作年限）等无关求职评估的信息。
   - "教育经历": (对象数组 [{{}}]) 包含所有教育背景。每个对象应包含学校、专业、学历、在校时间等字段。
   - "工作经历": (对象数组 [{{}}]) 包含所有工作经验。每个对象应包含公司名称、职位、在职时间、工作内容等字段。
   - "项目经历": (对象数组 [{{}}]) 包含所有项目经验。每个对象应包含项目名称、项目角色、项目时间、项目描述等字段。
   - "培训经历": (对象数组 [{{}}]) 包含所有培训经历。每个对象应包含培训机构、课程名称、时间等。
   - "个人评价": (字符串 "") 包含简历中的自我评价或职业总结部分。
   - "技能证书": (对象) 包含专业技能和获得的证书。可以按技能类别分类，例如 {{"语言技能": [], "专业技能": [], "证书": []}}。
   - 所有内容必须经过**信息浓缩与语言简化**，避免冗长重复，直击重点。
   - 不要思考,不要调用工具
3. **处理缺失信息**: 如果简历中没有找到某个类别的信息（例如，没有任何“培训经历”），请不要省略该键。对于期望是对象数组的键（如教育经历, 工作经历等），请使用空数组 [] 作为其值；对于期望是字符串的键（如个人评价），请使用空字符串 "" 作为其值；对于期望是对象的键（如基本信息），请使用空对象 {{}}。
</Instructions>
"""




# put_data(j)
#resumes_dict= [{str(i):put_data(j) for i,j in enumerate(resumes)}]
"""
chunk_size=20
for chunk in pd.read_csv("./db_ai.t_ck_cand_con0204_trunk.csv/db_ai.t_ck_cand_con0204_trunk.csv",
                             chunksize=chunk_size):
    resumes=list(chunk.iloc[:,2])
    break
resumes_dict= [{str(i):j for i,j in enumerate(resumes)}]

"""

from pymongo import MongoClient
client = MongoClient('mongodb://resume_db_test:F1rOLEw3OyRu558mROeX@dds-bp1f9e511ec921841.mongodb.rds.aliyuncs.com:3717/resume_db_test?authMechanism=SCRAM-SHA-1')
#print(client.list_database_names())
db = client['resume_db_test']
collection = db['resumeCollection']

"""
query = {
    "status": {"$in": [6]},
    "corpCode": "3DMed"
}

results = collection.find(query)
resumes={}
resumes_dict=[]
i=0
content_len=[]
for result in results:
    # resumeId,resumeContent
    i+=1
    if i>40:
        break
    #if i not in  [18]:
        #continue
    #if result['resumeId']!=403603:
       # continue
    content=result['resumeContent']
    content_len.append(len(content))
    resumes[result['resumeId']]=put_data(content)
    if i % 20 == 0:
        resumes_dict.append(resumes)
        resumes = {}
if resumes:
    resumes_dict.append(resumes)
"""
resumes=resumes[args.index:args.index+1]
resumes_dict=[]
if args.full:
    print('s')
    for t in range(10,args.end+1,10):
        resumes_={i:put_data(j) for i,j in enumerate(resumes*t)}
        resumes_dict.append(resumes_)
else:
    print('t')
    resumes_ = {i: put_data(j) for i, j in enumerate(resumes * args.end)}
    resumes_dict.append(resumes_)
content_len=len(resumes[0])


#print(resumes_dict)
# "/root/resume_summary/Qwen3-1.7B"
#"/root/resume_summary/qwen2.5-1.5B-merge"
async def send_request(session, resume):
    async with session.post('http://172.16.2.35:8009/summary', headers={'token':'d21b5ecb0ade56ca789a959e2bb57074'},
                            json=
                            {"text": resume,
                             "max_tokens": int(0.45*max(len(text) for text in list(resume.values()))),
                             "temperature": 0.0,
                             # "top_p": 1,
                             # "top_k":10,
                             #"presence_penalty": 0.4,
                             "instruct":True,
                             "model_path":"/root/resume_summary/Qwen3-1.7B",
                             "segment":False,
                             #"repetition_penalty":1.2,
                             "alter":False,   # True导致llm重新加载模型
                             # "max_num_seqs":60,  # 最大批次
                             "gpu_memory_utilization":0.9,
                             "enable_think":True
                             }) as response:
        print(f"Status: {response.status}")
        data = await response.json()
        #print("Response:", data)
        return data

async def main():
    connector = aiohttp.TCPConnector(limit_per_host=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        #tasks = [send_request(session, resume) for resume in resumes_dict]
        #results = await asyncio.gather(*tasks)
        if args.full:logger.info(f'-'*20)
        for i,resume in enumerate(resumes_dict):
            start = time.time()
            result=await send_request(session, resume)
            end = time.time()
            logger.info(f'耗时:{end - start}')
            # print(result)
            logger.info(f'{len(list(result.keys()))}个文本平均长度:{content_len}')
            if i==len(resumes_dict)-1:
                for key, value in result.items():
                    query = {
                        "status": {"$in": [6]},
                        "corpCode": "3DMed",
                        "resumeId": int(key)
                    }
                    all_ = []
                    value = extract_first_json(value)
                    try:
                        #results = collection.find(query)[0]
                        many_data = [
                            {'resumeId': int(key), 'resume_summary': value,'resume_txt':process_json1(value)}]
                        collection1.insert_many(many_data)
                        break
                    except Exception as e:
                        many_data = [{'resumeId': int(key), 'resume_summary': value}]
                        collection1.insert_many(many_data)
                        break


def process_json(all_,text):
    try:
        #logger.info(type(text))
        if isinstance(text,dict):
            for i in text.values():
                process_json(all_,i)
        elif isinstance(text,list):
            if text:
                for j in text:
                    process_json(all_,j)
        elif text:all_.append(text)
        return '\n'.join(all_)
    except Exception as e:
        #logger.error(f'{e},{text}')
        return text

def process_json1(text):
    try:
        if '</think>' in text:
            text_=text[::-1]
            text=re.findall(r'.*(?=>kniht/<)',text_,re.DOTALL)[0][::-1]
        all_=re.findall(r':\s*\n*"(?![\[{])(.*?)(?=")',text)
        if all_:return '\n'.join(all_)
        else:return text
    except Exception as e:
        logger.error(f'{e}')
        return text

def extract_first_json(text):
  if '```json' in text:
    pattern=r'```json\n*\s*(.*?)(?:$|`)'
  else:
    pattern = r'\{.*\}'
  try:
    match = re.findall(pattern,text, flags=re.DOTALL)[-1]
    return match
  except Exception as e:
    #print(f'提取JSON1出错:{e}:{text}')
    return str(text)

if __name__ == '__main__':
    collection1 = db['resume_summary']
    #start=time.time()
    result=asyncio.run(main())
    """
    end=time.time()
    logger.info(f'耗时:{end-start}')
    #print(result)

    for i in range(len(result)):
        for key,value in result[i].items():
            query = {
                "status": {"$in": [6]},
                "corpCode": "3DMed",
                "resumeId":int(key)
            }
            all_ = []
            value=extract_first_json(value)
            try:
                results = collection.find(query)[0]
                many_data = [{'resumeId':int(key),'resume':results['resumeContent'],'resume_summary':value}]
                collection1.insert_many(many_data)
            except Exception as e:
                many_data = [{'resumeId': int(key), 'resume': resumes[i], 'resume_summary': value}]
                collection1.insert_many(many_data)
    """

