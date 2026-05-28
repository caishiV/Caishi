from loguru import logger
import time
import os
import config

# 日志初始化
logger.add(
    os.path.join("./log", f"system_{time.strftime('%Y%m%d')}.log"),
    rotation="500 MB",
    encoding="utf-8",
    enqueue=True
)

def format_kg_data(kg_list: list) -> str:
    """将图谱三元组列表转为自然语言上下文"""
    if not kg_list:
        return "暂无相关参考知识"
    text = ""
    for item in kg_list:
        n1 = item.get("n.name", "")
        rel = item.get("type(r)", "")
        n2 = item.get("m.name", "")
        text += f"{n1}{rel}{n2}；"
    return text

def simple_text_check(text: str) -> bool:
    """简单输入合法性校验"""
    if not text or len(text.strip()) == 0:
        return False
    return True

class Timer:
    """简易计时器，统计推理/查询耗时"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.cost = round(self.end - self.start, 4)
        logger.info(f"执行耗时: {self.cost} s")