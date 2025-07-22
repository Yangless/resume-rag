import gc
import logging
import os
import time
import asyncio
import re
from typing import Optional
import torch
import json5
import pandas as pd
from fastapi import FastAPI, Header, HTTPException,Depends
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

token_init=os.getenv('TOKEN','d21b5ecb0ade56ca789a959e2bb57074')

logger = logging.getLogger("resume_summary_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_file = "resume_summary_logger.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def verify_token(token:str=Header(...)):
    try:
        if token.lower()!=token_init:
            raise HTTPException(status_code=401, detail=f"无效token")
        return {'token': token}
    except Exception as e:
        logger.error(f'token错误:{e}')

class Config:
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: int
    presence_penalty: Optional[float] = 0.0
    min_tokens: Optional[int] = 400
    instruct: bool
    segment: bool
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 0
    max_num_seqs:Optional[int] = None
    gpu_memory_utilization:Optional[float] = 0.9
    model_path: str=None
    alter: Optional[bool] = False
    enable_think:Optional[bool]=False


config=Config()

MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

#NUM_MODEL_COPIES = 2
#executor = ProcessPoolExecutor(max_workers=NUM_MODEL_COPIES)


llm = None

def split_and_merge_chunks(chunks, max_length=4000):
    try:
        merged_chunks = []
        current_chunk = ""
        for i,chunk in enumerate(chunks):
            if len(chunk)>max_length:
                t0=0
                for t in range(max_length,len(chunk),max_length):
                    if t0==0:
                        merged_chunks.append(chunk[:t])
                    else:
                        merged_chunks.append(chunk[t0:t])
                    t0 = t - 100
                continue

            if len(current_chunk) + len(chunk)  > max_length:
                merged_chunks.append(current_chunk)
                current_chunk = '\n'.join(chunks[i-3:i]) + chunk + "\n"
            else:
                if current_chunk:
                    current_chunk += "\n"
                current_chunk += chunk
        if len(current_chunk)>200:
            merged_chunks.append(current_chunk)
        else:
            if len(chunks[-6:])>50:
                test_200='\n'.join(chunks[-6:])
                merged_chunks.append(test_200)
        #if not merged_chunks:
            #merged_chunks.append(current_chunk)
        #print(merged_chunks)
        return merged_chunks
    except Exception as e:
        logger.error(f"模型拆分出错:{e}")
        return []

def load_vllm_model(config):
    global llm
    if llm is None :
        if config.max_num_seqs:
            llm = LLM(model=config.model_path, trust_remote_code=True, dtype="bfloat16",
                      tensor_parallel_size=1,
                      max_num_seqs=config.max_num_seqs,
                      gpu_memory_utilization=config.gpu_memory_utilization,
                      )
        else:llm = LLM(model=config.model_path, trust_remote_code=True, dtype="bfloat16",
                      tensor_parallel_size=1,
                      gpu_memory_utilization=config.gpu_memory_utilization,
                      )
    if config.alter:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        if config.max_num_seqs:
            llm = LLM(model=config.model_path, trust_remote_code=True, dtype="bfloat16",
                      tensor_parallel_size=1,
                      max_num_seqs=config.max_num_seqs,
                      gpu_memory_utilization=config.gpu_memory_utilization,
                      )
        else:
            llm = LLM(model=config.model_path, trust_remote_code=True, dtype="bfloat16",
                      tensor_parallel_size=1,
                      gpu_memory_utilization=config.gpu_memory_utilization,
                      )
    return llm


def extract_first_json(text):
    if '```json' in text:
        pattern=r'```json\n*\s*(.*?)[$|\n*`]'
    else:
        pattern = r'\{.*\}'
    try:
        match = re.findall(pattern, text, flags=re.DOTALL)[-1]
        return match
    except Exception as e:
        logger.info(f'提取JSON1出错:{e}:{text}')
        return str(text)

def process_json(text):
    label_list = ['基本信息', '教育经历', '工作经历', '项目经历', '培训经历', '个人评价', '技能证书']
    text_index = {}
    for i in label_list:
        try:
            text_index[i] = text.index(i)
        except Exception as e:
            pass
    try:
        text_dict = {}
        text_index = dict(sorted(text_index.items(), key=lambda x: x[1]))
        text_index_keys = list(text_index.keys())
        for i, key in enumerate(text_index_keys):
            index = int(text_index[key] + len(key))
            if i < len(text_index_keys) - 1:
                next_key = text_index_keys[i + 1]
                text_dict[key] = text[index:text_index[next_key]].replace(':','')
            else:
                text_dict[key] = text[index:].replace(':','')
        if not text_dict:
            return text
        #logger.info(f'JSON数据：{str(text_dict)}')
        return str(text_dict)
    except Exception as e:
        print(e)
        return text.replace(':','')


def diversity_ratio(s):
    unique_chars = len(set(s))
    total_chars = len(s)
    return unique_chars / total_chars

async def run_inference_on_gpu(resumes,sampling_params,config):
    try:
        llm = load_vllm_model(config)

        if not config.instruct:
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
            print(f"正在批量处理 {len(input_texts)} 条简历...")
            outputs = llm.generate(input_texts, sampling_params)

        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)

            batched_inputs = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": "你是一个助手，当被要求时只提供答案而不解释思考过程。"}
                        ,{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=config.enable_think,
                )
                for prompt in resumes
            ]

            outputs = llm.generate(batched_inputs, sampling_params)

        responses = [output.outputs[0].text for output in outputs]

        #logger.info(responses)
        #cleaned_responses = [extract_first_json(r) for r in responses]

        return responses
    except Exception as e:
        logger.error(f"推理错误: {e}")
        return []


class TextRequest(BaseModel):
    text:dict
    temperature:Optional[float]=1.0
    top_p:Optional[float]=1.0
    max_tokens:int
    presence_penalty:Optional[float]=0.0
    min_tokens:Optional[int]=400
    instruct:bool
    segment:bool
    model_path:str
    repetition_penalty:Optional[float]=1.0
    top_k:Optional[int]=0
    max_num_seqs: Optional[int] = None
    gpu_memory_utilization: Optional[float] = 0.9
    alter:Optional[bool]=False
    enable_think:Optional[bool] = False

app = FastAPI()


async def async_inference(input_texts,sampling_params,config):
    #loop = asyncio.get_event_loop()
    result = await run_inference_on_gpu(input_texts,sampling_params,config)
    return result
def merge_result(text):
    info, edu, work, project, train, evalu, skill = [], [], [], [], [], [], []
    merged={}
    else_info=[]

    for text_i in text:
        try:
            if not text_i:
                continue
            #if diversity_ratio(text_i)<0.1:
                #continue

            #print(text_i)
            text_i = re.sub(r'[\n\s]+', ' ', text_i)
            text_i=re.sub('，',',',text_i)
            try:
                text_dict = json5.loads(text_i)
            except Exception as e1:
                #else_info.append(str(text_i))
                #continue
                try:
                    text_i = re.sub(r'\[\\]', '\"\"', str(text_i))
                    text_i=re.sub(r'[][{}"]+', '', str(text_i))
                    text_i = re.sub(r'：', ':', text_i)
                    text_i_=process_json(text_i)
                    #text_i_ = re.sub(r":+\s*,", ":''", text_i_)
                    text_dict = json5.loads(text_i_)
                except Exception as e2:
                    print('重制JSON失败',e2)
                    else_info.append(str(text_i))
                    continue

            try:
                def safe_add(lst, val):
                    if val is not None:
                        lst.append(str(val))
                    elif isinstance(val, (str, int, float)):
                        lst.append(str(val))
                    elif isinstance(val, list):
                        lst.extend([str(v) for v in val if v is not None])
                safe_add(info, text_dict.get("基本信息",None))
                safe_add(edu, text_dict.get("教育经历",None))
                safe_add(work, text_dict.get("工作经历",None))
                safe_add(project, text_dict.get("项目经历",None))
                safe_add(train, text_dict.get("培训经历",None))
                safe_add(evalu, text_dict.get("个人评价",None))
                safe_add(skill, text_dict.get("技能证书",None))
            except Exception as e4:
                print(e4)

        except Exception as e3:
            logger.info(f"转换为JSON出错:{e3}:{text_i}")

    for name,resume_list in zip(["基本信息","教育经历","工作经历","项目经历","培训经历","个人评价","技能证书"],[info,edu,work,project,train,evalu,skill]):
        resume_list_=list(filter(None,resume_list))
        if resume_list_:
            merged[name]=resume_list_
    if else_info:
        merged['其他信息']=else_info
    merged=re.sub(r'[][{}\'"]+', '', str(merged))
    merged=re.sub(r'[,，\s]+',',',merged)
    return merged


@app.post("/summary")
async def analyze_text(data: TextRequest,token:str=Depends(verify_token)):
    try:
        config.temperature=data.temperature
        config.top_p=data.top_p
        config.max_tokens=data.max_tokens
        config.presence_penalty=data.presence_penalty
        config.min_tokens=data.min_tokens
        config.repetition_penalty=data.repetition_penalty
        config.instruct=data.instruct
        config.segment=data.segment
        config.top_k=data.top_k
        config.model_path = data.model_path
        config.alter=data.alter
        config.gpu_memory_utilization=data.gpu_memory_utilization
        config.enable_think=data.enable_think
        config.max_num_seqs=data.max_num_seqs
        #logger.info(model_path)
        ###
        data=data.text
        start = time.time()
        text_id=[]
        text=[]
        data_text = list(map(lambda x: x.replace('\\r', '\r').replace('：', ':').replace('\\n', '\n'), data.values()))
        if config.segment:
            for i,i_text in zip(data.keys(),data_text):
                #logger.info('s')
                i_text_list=i_text.splitlines()
                #logger.info(f'{i_text_list}')
                list_text=split_and_merge_chunks(i_text_list)
                list_text=list(filter(bool, list_text))
                #logger.info(f'原文:{list_text}')
                text_id.extend([i]*len(list_text))
                text.extend(list_text)
            config.max_tokens=int(0.45*max(len(i) for i in text))
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                presence_penalty=config.presence_penalty,
                min_tokens=config.min_tokens,
                repetition_penalty=config.repetition_penalty,
                top_k=config.top_k

            )
            result = await async_inference(text,sampling_params,config)
            #logger.info(f'模型输出:{result}')

            df=pd.DataFrame(zip(text_id,result),columns=['id','text'])
            #print(df)
            df_merge = (
                df.groupby('id', as_index=False,sort=False)['text']
                .apply(lambda x: merge_result(x))
                .reset_index(drop=True)
            )
            df_merge.columns = ['id', 'merged_info']
            #print(df_merge)
            result_end=df_merge.set_index('id')['merged_info'].to_dict()
            end = time.time()
            logger.info(f"耗时：{end - start}" )
            return result_end
        else:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                presence_penalty=config.presence_penalty,
                min_tokens=config.min_tokens,
                repetition_penalty=config.repetition_penalty,
                top_k=config.top_k

            )
            result = await async_inference(data_text, sampling_params,config)
            result_end={key:value for key,value in zip(data.keys(),result)}
            end = time.time()
            logger.info(f"耗时：{end - start}")
            return result_end

    except Exception as e:
        logger.error(f'模型推理出错: {e}')
        return {"error": str(e)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app="test_batch:app",
        host="0.0.0.0",
        port=8009,
        reload=False,
        workers=1
    )