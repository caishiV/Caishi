import torch
import os

# ===================== 服务基础配置 =====================
HOST = "0.0.0.0"
PORT = 8000
SERVER_RELOAD = True

# ===================== Neo4j 知识图谱配置 =====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456"
KG_LIMIT = 5  # 图谱查询返回最大条数

# 图谱领域标签
KG_TYPE_GENERAL = "general"    # 通用常识图谱
KG_TYPE_TECH = "technology"    # 技术领域图谱

# ===================== 大模型配置 =====================
BASE_MODEL_DIR = "./base_model"
LORA_SAVE_DIR = "./lora_models"
MODEL_QUANT_BITS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 生成参数
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.7

# LoRA 训练参数
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ===================== 数据回流配置 =====================
BACKFLOW_DATA_DIR = "./backflow_data"
BACKFLOW_LOG_NAME = "interaction_log.csv"

# ===================== 目录自动创建 =====================
def init_dirs():
    dir_list = ["./log", LORA_SAVE_DIR, BACKFLOW_DATA_DIR]
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

# 初始化目录
init_dirs()