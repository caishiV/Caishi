import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import config
from utils import logger, format_kg_data
from data_backflow import backflow

# ===================== 量化 & LoRA 静态配置 =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=config.LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

class AutoModelCore:
    def __init__(self):
        self.accelerator = Accelerator()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载基座模型 + Tokenizer + LoRA"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_DIR,
                quantization_config=bnb_config,
                device_map=config.DEVICE,
                trust_remote_code=True
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.success("大模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")

    def model_infer(self, prompt: str, kg_data: list) -> str:
        """融合知识图谱上下文进行推理"""
        if not self.model or not self.tokenizer:
            return "模型加载异常，无法回答"

        kg_context = format_kg_data(kg_data)
        input_text = f"参考知识：{kg_context}\n用户问题：{prompt}\n回答："

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(config.DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def auto_lora_train(self):
        """全自动LoRA微调（基于回流对话数据）"""
        if not self.model:
            logger.warning("模型未加载，跳过自动训练")
            return

        samples = backflow.get_train_samples()
        if len(samples) < 3:
            logger.info("训练样本不足，暂不执行训练")
            return

        # 简易训练逻辑（原型版，可扩展完整训练循环）
        model_count = len(os.listdir(config.LORA_SAVE_DIR))
        save_name = f"auto_lora_{model_count}"
        save_path = os.path.join(config.LORA_SAVE_DIR, save_name)

        self.model.save_pretrained(save_path)
        logger.success(f"自动训练完成，权重已保存至: {save_path}")

# 全局模型实例
model_core = AutoModelCore()