import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

class Config:
    model_path = "./qwen2.5-1.5B"
    output_dir = "./qwen2.5-resume-finetune"
    train_path = "./train.csv"
    eval_path = "./eval.csv"

    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    # 训练配置
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_epochs = 3
    warmup_steps = 100
    logging_steps = 1000
    save_steps = 1000
    max_length = 1024
    fp16 = True
    gradient_checkpointing = True


config = Config()

class ResumeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length


        self.data = pd.read_csv(data_path,encoding='utf-8')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        resume_text = row['resume_text']
        target_summary = row['summary']

        # 构建输入文本
        input_text = f"""从下面这段简历文本中提取涉及以下7类标签的总结：基本信息,教育经历,工作经历,项目经历,培训经历,个人评价,技能证书。
        每份工作或项目经历字数不超过100字;如果标签没有提取到内容则返回空字符串;不要嵌套JSON。
简历文本:
{resume_text}

返回JSON格式:
```json
{{"基本信息":"","教育经历":[],"工作经历":[],"项目经历":[],"培训经历":[],"个人评价":"","技能证书":""}}
```
"""

        # 完整的目标文本
        full_text = input_text + target_summary
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_only = self.tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        input_length = len(input_only["input_ids"])

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        labels = input_ids.clone()
        labels[:input_length] = -100  # 屏蔽 prompt 部分的 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class DataCollatorForChatGLM(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        return batch


class ChatGLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def setup_model_and_tokenizer():
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,

    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        load_in_8bit=False,
        #attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        #device_map="auto"
    )
    model.enable_input_require_grads()

    # 启用梯度检查点
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none"
    )
    #import os
    #os.environ["BITSANDBYTES_DISABLE_8BIT"] = "1"

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train():

    model, tokenizer = setup_model_and_tokenizer()

    print("准备数据集...")
    train_dataset = ResumeDataset(config.train_path, tokenizer, config.max_length)
    eval_dataset = ResumeDataset(config.eval_path, tokenizer, config.max_length)

    data_collator = DataCollatorForChatGLM(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        max_length=config.max_length,
        return_tensors="pt"
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        fp16=config.fp16,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=config.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # 禁用wandb等日志工具
        #deepspeed="deepspeed_config.json"
    )

    # 创建训练器
    trainer = ChatGLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

def load_and_test():
    print("加载微调后的模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, config.output_dir)
    model = model.merge_and_unload()  # 合并LoRA权重

    tokenizer = AutoTokenizer.from_pretrained(
        config.output_dir,
        trust_remote_code=True
    )
    test_resume = """
李四
联系方式：北京市朝阳区，手机：13800138000
邮箱：lisi@test.com

教育经历：
北京大学 计算机科学 本科 2015-2019

工作经验：
腾讯 高级开发工程师 2019-2023
- 负责微信支付系统开发
- 优化了支付流程，提升用户体验

技能专长：
Go语言、Redis、MySQL
    """

    input_text = f"""从下面这段简历文本中提取涉及以下7类标签的总结：基本信息,教育经历,工作经历,项目经历,培训经历,个人评价,技能证书。
        每份工作或项目经历字数不超过100字;如果标签没有提取到内容则返回空字符串;不要嵌套JSON。
简历文本:
{test_resume}

返回JSON格式:
```json
{{"基本信息":"","教育经历":[],"工作经历":[],"项目经历":[],"培训经历":[],"个人评价":"","技能证书":""}}
```
"""

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2000,
            do_sample=True,
            top_p=0.7,
            temperature=0.8,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #summary_start = response.find("提取的信息如下：") + len("提取的信息如下：")
    #summary = response[summary_start:].strip()

    print("微调后的提取结果:")
    print(response)


if __name__ == "__main__":
    train()
    #load_and_test()