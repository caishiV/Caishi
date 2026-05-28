from py2neo import Graph, Node, Relationship
from py2neo.errors import AuthError, ServiceUnavailable
import config
from utils import logger

class KnowledgeGraphEngine:
    def __init__(self):
        try:
            self.graph = Graph(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            logger.success("知识图谱引擎连接成功")
        except (AuthError, ServiceUnavailable):
            logger.error("Neo4j 连接失败，请检查服务、账号密码与端口")
            self.graph = None

    def auto_select_kg_type(self, query: str) -> str:
        """规则引擎自动选择图谱领域（可替换为大模型分类）"""
        tech_words = {"Python", "代码", "模型", "算法", "训练", "开发", "编程"}
        query_lower = query.lower()
        for word in tech_words:
            if word in query_lower:
                return config.KG_TYPE_TECH
        return config.KG_TYPE_GENERAL

    def query_kg(self, query_text: str) -> list:
        """根据文本自动选库并查询三元组"""
        if not self.graph:
            return []
        kg_type = self.auto_select_kg_type(query_text)
        cypher = f"""
        MATCH (n:{kg_type})-[r]->(m)
        WHERE n.name CONTAINS $kw
        RETURN n.name, type(r), m.name
        LIMIT $limit
        """
        try:
            res = self.graph.run(
                cypher,
                kw=query_text,
                limit=config.KG_LIMIT
            ).data()
            return res
        except Exception as e:
            logger.error(f"图谱查询异常: {str(e)}")
            return []

    def add_kg_entity(self, entity_a: str, relation: str, entity_b: str, kg_type: str):
        """自动新增实体与关系（图谱自演化）"""
        if not self.graph:
            return False
        node_a = Node(kg_type, name=entity_a)
        node_b = Node(kg_type, name=entity_b)
        rel = Relationship(node_a, relation, node_b)
        self.graph.merge(node_a)
        self.graph.merge(node_b)
        self.graph.merge(rel)
        logger.info(f"图谱新增知识: {entity_a} ->{relation}-> {entity_b}")
        return True

# 全局单例
kg_engine = KnowledgeGraphEngine()