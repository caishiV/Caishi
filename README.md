# Caishi人形AI自治系统 README
 
Humanoid AI + Auto Model R&D + Autonomous Knowledge Graph System
结合人形交互、AI自动模型研发、知识图谱自主选型演化的全链路自治AI原型，适配8G内存本地离线部署，基于 Python + PyTorch + Neo4j 实现。
 
 
 
一、项目简介
 
本项目构建一套三层自治AI体系：
 
1. 人形交互层：模拟物理/数字人形终端，负责多模态输入、指令解析、结果输出
2. 知识图谱自治层：AI自动识别场景、选择对应图谱库、检索知识、自主更新实体关系
3. 模型自动研发层：基于LoRA轻量化微调，实现模型全自动训练、迭代、版本保存
 
核心特性：
 
- 本地离线运行，数据不上云，隐私可控
- 知识图谱作为事实锚点，有效降低大模型幻觉
- 交互数据自动回流，形成「交互→推理→数据回流→模型自训练→能力升级」闭环
- 4bit模型量化，最低8G内存、4G显存即可流畅运行
 
 
 
二、目录结构
 
plaintext  
ai_autonomous_system/
├── config.py            # 全局配置项、路径、账号、超参
├── utils.py             # 公共工具、日志、数据格式化、计时器
├── kg_engine.py         # 知识图谱引擎：选型/查询/新增实体
├── humanoid_io.py       # 人形终端IO：输入解析、WebSocket输出
├── data_backflow.py     # 数据回流：交互日志存储、训练样本读取
├── auto_model.py        # 模型加载、推理、全自动LoRA训练
├── main_server.py       # 项目主入口、FastAPI+WebSocket服务
├── requirements.txt     # 环境依赖清单
├── README.md            # 项目说明文档
├── log/                 # 系统运行日志
├── lora_models/         # 自动训练生成的LoRA权重
├── backflow_data/       # 用户交互回流数据集
└── base_model/          # 存放轻量化基座大模型
 
 
 
 
三、环境与硬件要求
 
1. 最低硬件配置
 
- 系统：Windows 10/11 / Linux
- 内存：≥ 8GB
- 显存：≥ 4GB（支持CUDA加速）
- 磁盘：剩余空间 ≥ 20GB
 
2. 软件版本
 
- Python：3.10
- PyTorch：2.1.x (CUDA 11.8)
- Neo4j：5.x 社区版（本地单机）
 
 
 
四、快速部署步骤
 
步骤1：安装依赖
 
1. 新建虚拟环境（推荐）
 
bash  
python -m venv venv
# Windows 激活环境
venv\Scripts\activate
 
 
2. 批量安装依赖
 
bash  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
 
 
步骤2：部署并初始化 Neo4j
 
1. 本地安装 Neo4j 社区版，启动服务
2. 访问后台： http://localhost:7474 
3. 修改初始密码为： 123456 （与 config.py 保持一致）
4. 在Neo4j执行以下Cypher语句，导入测试图谱数据：
 
cypher  
// 通用常识图谱
CREATE (g1:general{name:"地球"})
CREATE (g2:general{name:"人类"})
CREATE (g1)-[r1:拥有]->(g2)

// 技术领域图谱
CREATE (t1:technology{name:"Python"})
CREATE (t2:technology{name:"编程语言"})
CREATE (t1)-[r3:属于]->(t2)
CREATE (t3:technology{name:"大模型"})
CREATE (t1)-[r4:可开发]->(t3)
 
 
步骤3：准备基座大模型
 
将轻量化开源大模型（如 Qwen-1.8B、Llama2 量化版等）完整放入  ./base_model  文件夹。
 
要求：文件夹内包含模型权重、配置文件、分词器文件。
 
步骤4：启动系统服务
 
bash  
python main_server.py
 
 
服务启动成功后：
 
- 服务地址： ws://127.0.0.1:8000/ws/humanoid 
 
 
 
五、功能使用说明
 
1. 交互测试
 
使用 WebSocket 调试工具（Postman、在线WS工具等）连接上述地址，发送文本消息：
 
- 发送  地球上有什么  → 自动匹配通用常识图谱
- 发送  Python 能做什么  → 自动匹配技术领域图谱
 
2. 核心运行流程
 
plaintext  
用户输入 → 人形终端解析任务
     ↓
知识图谱自动选型 + 知识检索
     ↓
大模型融合图谱知识推理回答
     ↓
结果推送至人形终端输出
     ↓
交互数据自动回流存储
     ↓
触发全自动LoRA微调 → 模型迭代升级（闭环）
 
 
3. 独立模块调试
 
测试知识图谱
 
python  
from kg_engine import kg_engine
print(kg_engine.query_kg("Python"))
# 新增知识
kg_engine.add_kg_entity("人工智能","使用","Python","technology")
 
 
测试模型推理
 
python  
from auto_model import model_core
from kg_engine import kg_engine
kg_data = kg_engine.query_kg("地球")
res = model_core.model_infer("地球上有什么", kg_data)
print(res)
 
 
测试数据回流
 
python  
from data_backflow import backflow
backflow.save_interaction("测试问题","测试回答","general")
print(backflow.get_train_samples())
 
 
 
 
六、关键配置说明（config.py）
 
可根据硬件、业务自行修改：
 
1. 服务端口： HOST / PORT 
2. Neo4j 连接： NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 
3. 模型量化： MODEL_QUANT_BITS  默认4bit，适配低显存
4. 生成参数： MAX_NEW_TOKENS  最大生成长度、 TEMPERATURE  随机度
5. LoRA训练： LORA_R / LORA_ALPHA  微调超参
6. 目录路径：模型、日志、回流数据存储路径
 
 
 
七、常见问题 & 排错指南
 
1. CUDA out of memory 显存溢出
 
- 确认已开启  4bit  量化
- 关闭其他占用GPU的程序
- 选用参数更小的基座模型
 
2. Neo4j 连接失败 / 认证错误
 
- 检查Neo4j服务是否正常启动
- 核对账号、密码、端口与  config.py  一致
- 关闭本地防火墙/端口占用程序
 
3. 模型加载失败
 
-  base_model  目录必须存放完整模型文件
- 检查PyTorch、transformers版本是否匹配
 
4. WebSocket 连接失败
 
- 确认8000端口未被占用
- 本地访问使用  127.0.0.1 ，勿使用外网IP
 
5. 自动训练不执行
 
- 回流数据样本数量 ≥3 才会触发训练（代码内阈值）
- 检查  backflow_data  目录是否有交互日志
 
 
 
八、模块扩展方向
 
1. 知识图谱模块- 将关键词匹配选型升级为大模型语义分类，提升领域识别精度
- 增加多图谱融合、跨领域推理、知识消歧、自动去重
2. 模型训练模块- 完善完整训练循环、学习率调度、Batch加载、早停策略
- 支持增量训练、模型版本管理、自动灰度切换
3. 人形交互模块- 接入 OpenCV + 摄像头、PyAudio + 麦克风，实现真实多模态交互
- 对接物理人形机器人舵机/动作控制接口
4. 系统能力扩展- 增加内容风控、违规检测模块
- 接入流式实时知识图谱，支持资讯、动态数据更新
- 增加前端可视化控制台，监控模型、图谱、日志状态
 
 
 
九、技术栈
 
- 后端服务：FastAPI + WebSocket
- 大模型框架：PyTorch + Transformers + PEFT + bitsandbytes
- 知识图谱：Neo4j + py2neo
- 日志工具：loguru
- 数据处理：Pandas / CSV
 
 
 
十、开源协议
 
本项目为技术原型Demo，仅供学习、研究、二次开发使用。