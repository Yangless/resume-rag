import logging
import os
import time
import asyncio
import re
import json5
import pandas as pd
from fastapi import FastAPI, Header, HTTPException,Depends
from pydantic import BaseModel
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
    model_path = "qwen2.5-1.5B"
    output_dir = "qwen2.5-resume-finetune1"

config = Config()

MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

#NUM_MODEL_COPIES = 2
#executor = ProcessPoolExecutor(max_workers=NUM_MODEL_COPIES)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=600,
    #stop=["```"]  # 如果输出包含 ```json ... ```，可以提前停止
)

llm = None

def split_and_merge_chunks(chunks, max_length=1024):
    try:
        merged_chunks = []
        current_chunk = ""
        for i,chunk in enumerate(chunks):
            if len(current_chunk) + len(chunk) + 1 > max_length:
                merged_chunks.append(current_chunk)
                current_chunk = '\n'.join(chunks[i-3:i]) + chunk + "\n"
            else:
                if current_chunk:
                    current_chunk += "\n"
                current_chunk += chunk
        if len(current_chunk)>200:
            merged_chunks.append(current_chunk)
        else:
            test_200='\n'.join(chunks[-15:])
            merged_chunks.append(test_200)
        if not merged_chunks:
            merged_chunks.append(current_chunk)
        return merged_chunks
    except Exception as e:
        logger.error(f"模型拆分出错:{e}")
        return []

def load_vllm_model():
    global llm
    if llm is None:
        llm = LLM(model="qwen2.5-1.5B-merge1", trust_remote_code=True,
                  dtype="half", tensor_parallel_size=1)
    return llm


def extract_first_json(text):
    pattern = r'{.*?}'
    try:
        match = re.findall(pattern, text, flags=re.DOTALL)[-1]
        return match
    except Exception as e:
        #logger.error(f'提取JSON1出错:{e}')
        pattern = r'{(.*)'
        try:
            match = re.findall(pattern, text, flags=re.DOTALL)[-1]
            return match
        except Exception as e:
            logger.info(f'提取JSON2出错:{e}', text)
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
                text_dict[key] = text[index:text_index[next_key]]
            else:
                text_dict[key] = text[index:]
        if not text_dict:
            return text
        return text_dict
    except Exception as e:
        print(e)
        return text


def diversity_ratio(s):
    unique_chars = len(set(s))
    total_chars = len(s)
    return unique_chars / total_chars

async def run_inference_on_gpu(resumes):
    try:
        llm = load_vllm_model()

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

        responses = [output.outputs[0].text for output in outputs]
        cleaned_responses = [extract_first_json(r) for r in responses]
        return cleaned_responses
    except Exception as e:
        logger.error(f"推理错误: {e}")
        return []


class TextRequest(BaseModel):
    text:dict

app = FastAPI()


async def async_inference(input_texts):
    #loop = asyncio.get_event_loop()
    result = await run_inference_on_gpu(input_texts)
    return result
def merge_result(text):
    info, edu, work, project, train, evalu, skill = [], [], [], [], [], [], []
    merged={}
    else_info=[]

    for text_i in text:
        try:
            if not text_i:
                continue
            if diversity_ratio(text_i)<0.1:
                continue

            #print(text_i)
            text_i = re.sub(r'[\n\s]+', ' ', text_i)
            text_i=re.sub('，',',',text_i)
            try:
                text_dict = json5.loads(text_i)
            except Exception as e:
                #else_info.append(str(text_i))
                #continue
                text_i=re.sub(r'[][{}"]+', '', str(text_i))
                text_i = re.sub(r'：', ':', text_i)
                text_i_=process_json(text_i)
                text_i_ = re.sub(r":,", "''", text_i_)
                try:
                    text_dict = json5.loads(text_i_)
                except Exception as e:
                    #print(text_i_)
                    print('重制JSON失败',e)
                    else_info.append(str(text_i))
                    continue

            def safe_add(lst, val):
                if isinstance(val, (str, int, float)):
                    lst.append(str(val))
                elif isinstance(val, list):
                    lst.extend([str(v) for v in val if v is not None])
                elif val is not None:
                    lst.append(str(val))

            safe_add(info, text_dict.get("基本信息"))
            safe_add(edu, text_dict.get("教育经历"))
            safe_add(work, text_dict.get("工作经历"))
            safe_add(project, text_dict.get("项目经历"))
            safe_add(train, text_dict.get("培训经历"))
            safe_add(evalu, text_dict.get("个人评价"))
            safe_add(skill, text_dict.get("技能证书"))

        except Exception as e:
            logger.info(f"转换为JSON出错:{e}:{text}")

    for name,resume_list in zip(["基本信息","教育经历","工作经历","项目经历","培训经历","个人评价","技能证书"],[info,edu,work,project,train,evalu,skill]):
        resume_list_=list(filter(None,resume_list))
        if resume_list_:
            merged[name]=resume_list_
    if else_info:
        merged['其他信息']=else_info
    return re.sub(r'[][{}\'"]+', '', str(merged))


@app.post("/summary")
async def analyze_text(data: TextRequest,token:str=Depends(verify_token)):
    try:
        data=data.text
        start = time.time()
        text_id=[]
        text=[]
        data_text = list(map(lambda x: x.replace('\\r', '\r').replace('：', ':').replace('\\n', '\n'), data.values()))
        for i,i_text in zip(data.keys(),data_text):
            i_text_list=i_text.splitlines()
            list_text=split_and_merge_chunks(i_text_list)
            text_id.extend([i]*len(list_text))
            text.extend(list_text)
        result = await async_inference(text)
        #print(result)

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