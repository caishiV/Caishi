import csv
import os
import time
import config
from utils import logger

class DataBackflow:
    def __init__(self):
        self.file_path = os.path.join(config.BACKFLOW_DATA_DIR, config.BACKFLOW_LOG_NAME)
        self._init_csv()

    def _init_csv(self):
        """初始化csv表头"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "query", "answer", "domain"])

    def save_interaction(self, query: str, answer: str, domain: str):
        """保存单条交互数据"""
        now_time = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.file_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([now_time, query, answer, domain])
            logger.debug("交互数据已回流保存")
        except Exception as e:
            logger.error(f"数据保存失败: {str(e)}")

    def get_train_samples(self) -> list:
        """读取回流数据，返回训练样本列表"""
        samples = []
        if not os.path.exists(self.file_path):
            return samples
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "instruction": row["query"],
                    "output": row["answer"]
                })
        return samples

# 全局实例
backflow = DataBackflow()