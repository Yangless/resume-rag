from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型和分词器
model_name = "qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载LoRA模型
lora_model_path = "qwen2.5-resume-finetune1"
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload()

merged_model_path = "qwen2.5-1.5B-merge1"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)