from fastapi import WebSocket
import config
from kg_engine import kg_engine
from utils import logger, simple_text_check

class HumanoidIO:
    @staticmethod
    def parse_user_input(raw_input: str) -> dict:
        """解析原始输入，生成标准化任务结构体"""
        if not simple_text_check(raw_input):
            return {"valid": False, "task_text": "", "domain": ""}

        domain = kg_engine.auto_select_kg_type(raw_input)
        task = {
            "valid": True,
            "task_text": raw_input.strip(),
            "domain": domain
        }
        logger.info(f"解析任务: {task}")
        return task

    @staticmethod
    async def ws_send_result(ws: WebSocket, content: str):
        """WebSocket 实时推送结果（模拟语音/屏幕/肢体反馈）"""
        try:
            await ws.send_text(content)
        except Exception as e:
            logger.error(f"结果推送失败: {str(e)}")

# 全局实例
human_io = HumanoidIO()