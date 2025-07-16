import logging
import os
import time
import asyncio
import re
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

# class Config:
#     model_path = "/root/resume_summary/Qwen3-1.7B/"
#     output_dir = "Qwen3-1.7B-resume"
#
# config = Config()

MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# sampling_params = SamplingParams(
#     temperature=0.7,  # 控制模型输出的随机性。0.01是一个非常低的值，意味着模型会非常确定地选择概率最高的词语，输出会非常保守和连贯，但几乎没有创造性，接近贪婪搜索。
#     top_p=1,  # 从累积概率超过1.0的最小词汇集合中进行采样。当temperature极低时，top_p参数的影响微乎其微，设为1.0表示不限制采样范围。
#     # top_k=-1,  # 限制模型在生成下一个词时只考虑概率最高的k个词。设置为-1表示不使用top_k限制，即考虑所有词汇，或者由top_p来控制。
#     # min_p=0,  # 这是一个较少见的参数，它会过滤掉概率低于min_p的词。设置为0表示不进行此过滤，所有词都有被选择的机会（除非被top_k或top_p限制）。
#     # repetition_penalty=1.2,  # 对已经生成的词进行惩罚，使其再次出现的概率降低。1.15是一个温和的惩罚值，有助于减少文本中的重复内容，使摘要更自然。
#     # presence_penalty=0.4,
#     max_tokens=300,  # 限制生成文本的最大长度（以token计）。当生成文本达到此长度时，模型将停止输出。
#     # n=1,  # 指定生成多少个不同的输出序列。设置为1表示只生成一个最佳的输出结果。
# # Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("/root/resume_summary/Qwen3-1.7B")

# Configurae the sampling parameters (for thinking mode)
# sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
# do_sample=True (非常重要，开启采样)
# temperature=0.7 (适度随机性)
# top_p=0.9 (选择高质量的候选词)
# repetition_penalty=1.2 (开始惩罚重复)
# max_new_tokens：根据你的需求设置一个合理上限。
# )

llm = None

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/resume_summary/Qwen3-1.7B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
# llm = LLM(model="/root/resume_summary/Qwen3-1.7B")



def load_vllm_model():
    global llm
    # if llm is None:
        #/root/resume_summary/Qwen3-4B
        #/root/resume_summary/Qwen3-1.7B
        #/root/resume_summary/Qwen3-0.6B
        #/root/resume_summary/SmolLM3-3B-unsloth-bnb-4bit
        # /root/resume_summary/Qwen3-4B-AWQ
        # /root/resume_summary/Qwen3-4B
        # /root/resume_summary/SmolLM3-3B
        # llm = LLM(model="/root/resume_summary/Qwen3-1.7B", trust_remote_code=True,
        #           dtype="auto", tensor_parallel_size=1)
    return llm


async def run_inference_on_gpu(resumes):
    try:
        # llm = load_vllm_model()
        llm = LLM(model="/root/resume_summary/Qwen3-1.7B")
        input_texts = []
        for resume in resumes:
            # Prepare the input to the model



            # messages = [
            #     {"role": "user", "content": prompt}
            # ]
            input_text = f""" 请你作为专业的简历信息提取助手。不要输出无意义重复的数字或者字母。
            从提供的简历文本中精准提取并分类以下7类信息：基本信息、教育经历、工作经历、项目经历、培训经历、个人评价、技能证书。
            提取规则和格式要求：
            返回格式严格为JSON对象。
            不要输出无关提示信息。
            只返回JSON对象，不要反其他任何信息
            JSON对象必须且仅包含以下7个顶级键：

                基本信息 (字符串): 包含姓名、性别、年龄、工作年限、学历、现居地、户籍、政治面貌、求职状态、求职意向（职位、薪资、城市、行业、类型）等。所有这些信息应合并为一个简洁的字符串。
                教育经历 (字符串数组): 每一项应是该段教育经历的总结。例如："时间段 学校名称 专业 学历 学习类型"。
                工作经历 (字符串数组): 每一项应是该段工作经历的总结，字数不超过100字。例如："时间段 公司名称 职位 职责描述"。
                项目经历 (字符串数组): 每一项应是该段项目经历的总结，字数不超过100字。例如："时间段 项目名称 角色 职责描述 成果"。
                培训经历 (字符串数组): 每一项应是该段培训经历的总结。
                个人评价 (字符串): 包含对自身的总结和评价。
                技能证书 (字符串): 包含掌握的技能和获得的证书。
                内容归类： 简历中的所有相关信息都应归入这7个指定类别中。不允许创建其他键。
                
                空标签处理： 如果某个标签在简历中没有提取到任何内容，请返回其对应的空字符串（如 ""）或空列表（如 []）。
                
                避免重复信息： 提取的信息应精炼，避免不必要的重复。
                
                避免JSON嵌套： 除了顶级的JSON对象，其内部的值（字符串或字符串数组）不应包含额外的JSON结构。
            
                简历文本:
                {resume}
            
                返回JSON格式:
                ```json
                {{"基本信息":"","教育经历":[],"工作经历":[],"项目经历":[],"培训经历":[],"个人评价":"","技能证书":""}}
                ```
                
                """


            messages = [
                {"role": "user", "content": input_text}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Set to False to strictly disable thinking
            )

            # # Generate outputs
            # outputs = llm.generate([text], sampling_params)
            #
            # # Print the outputs.
            # for output in outputs:
            #     prompt = output.prompt
            #     generated_text = output.outputs[0].text
            #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            # # input_texts_dict = {"role": "user", "content": input_text}
            input_texts.append(text)

        print(f"正在批量处理 {len(input_texts)} 条简历...")
        print(input_texts)
        outputs = llm.generate(input_texts, sampling_params)
        print("outputtttttt",outputs)

        responses = [output.outputs[0].text for output in outputs]
        # print("responsesssssssss", responses)
        return responses
    except Exception as e:
        logger.error(f"推理错误: {e}")
        return []


class TextRequest(BaseModel):
    text:dict

app = FastAPI()


async def async_inference(input_texts):
    result = await run_inference_on_gpu(input_texts)
    return result


@app.post("/summary")
async def analyze_text(data: TextRequest,token:str=Depends(verify_token)):
    try:
        data=data.text
        start = time.time()
        text_id=[]
        text=[]
        data_text = list(map(lambda x: x.replace('\\r', '\r').replace('：', ':').replace('\\n', '\n'), data.values()))
        for i,i_text in zip(data.keys(),data_text):
            text_id.append(i)
            text.append(i_text)

        # print("texttttttt",text)
        result = await async_inference(text)
        
        result_dict = {}
        for i, res in zip(text_id, result):
            result_dict[i] = res

        end = time.time()
        logger.info(f"耗时：{end - start}" )
        return result_dict
    except Exception as e:
        logger.error(f'模型推理出错: {e}')
        return {"error": str(e)}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app="resume_parser:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
        workers=1
    )
