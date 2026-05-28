from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import config
from humanoid_io import human_io
from kg_engine import kg_engine
from auto_model import model_core
from data_backflow import backflow
from utils import logger, Timer

app = FastAPI(title="人形AI自治系统", version="1.0")

@app.websocket("/ws/humanoid")
async def humanoid_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("新人形终端连接建立")

    try:
        while True:
            # 1. 接收用户输入
            raw_msg = await websocket.receive_text()

            # 2. 解析任务
            task = human_io.parse_user_input(raw_msg)
            if not task["valid"]:
                await human_io.ws_send_result(websocket, "输入内容无效，请重新提问")
                continue

            query = task["task_text"]
            domain = task["domain"]

            # 3. 知识图谱查询
            with Timer():
                kg_res = kg_engine.query_kg(query)

            # 4. 模型推理
            with Timer():
                answer = model_core.model_infer(query, kg_res)

            # 5. 推送结果到人形终端
            await human_io.ws_send_result(websocket, answer)

            # 6. 数据回流持久化
            backflow.save_interaction(query, answer, domain)

            # 7. 触发自动训练
            model_core.auto_lora_train()

    except WebSocketDisconnect:
        logger.warning("人形终端连接断开")
    except Exception as e:
        logger.error(f"服务异常: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main_server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.SERVER_RELOAD
    )