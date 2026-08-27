"""
智能客服问答服务

处理流程：用户口语提问
-> 查询改写（LLM）：大白话 -> 规范检索关键词
-> 阈值检索：相似度 < similarity_threshold 的文档被过滤
-> 检索结果为空 -> 返回兜底文案，不硬答
-> 生成（带历史）：原问题 + 参考资料 -> 简洁回答

依赖：vector_stores.py（检索）、file_history_store.py（历史）、config_data.py（参数）、.env（API key）
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from dotenv import load_dotenv

# .env 在项目根目录（RAG项目 的上一级），用绝对路径加载，不受启动目录影响
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from embeddings import get_embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import config_data as config
from vector_stores import VectorStoreService
from file_history_store import get_history


class RagService(object):
    def __init__(self):
        # 1. 向量检索器
        # 共享单例：bge-m3 模型全应用只加载一次
        self.retriever = VectorStoreService(get_embeddings()).get_retriever()

        self.chat_model = ChatOpenAI(
            model="deepseek-v4-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            temperature=0.7,
            max_tokens=2048,
            timeout=60,
        )

        # ---- 环节一：查询改写链 ----
        # 把口语化问题改写成适合向量检索的关键词，只输出改写结果
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是服装电商的查询改写助手。把用户口语化的问题改写成适合向量检索的关键词短语，"
                "只输出改写结果本身，不要解释，不要加标点和换行。"),
                ("user", "{input}"),
            ]
        )
        self.rewrite_chain = self.rewrite_prompt | self.chat_model | StrOutputParser()

        # ---- 环节二：意图追问链 ）----
        # 独立成链的原因：一条链一个职责；代码能读 CLARIFY: 前缀做轮数判环；
        # 追问轮直接 return，不发起检索。判断依据用"原始问题"而非改写结果
        # （改写是检索用的关键词压缩，会把模糊问题压成看似信息齐全的短语）。
        self.clarify_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是服装电商智能客服的\"意图判断\"助手。你的唯一任务：判断用户问题是否需要追问，"
                "绝不回答用户的购物问题本身。\n\n"
                "需要追问：用户提出了具体的购物需求（如搭配、推荐、选购衣服/裤子/鞋等），"
                "但缺少回答所必需的关键信息。\n"
                "关键信息包括：性别、身高、体重、穿着场合（上班/运动/约会/日常等）、"
                "风格偏好（休闲/商务/甜美/运动风等）、具体需求类型（如连衣裙、裤子、羽绒服）。\n\n"
                "不需要追问（输出 PASS）：\n"
                "1. 闲聊、问候、感谢、情绪表达：如\"你好\"\"谢谢\"\"在吗\"\n"
                "2. 问题已包含足够信息（已明确身高体重、性别、场合或具体商品）\n"
                "3. 与服装购物无关的问题\n"
                "4. 历史对话中助手最近已经追问过（消息以 CLARIFY: 开头），本轮不再追问，把回答交给主流程\n\n"
                "输出格式（严格，只输出一行，不要任何其他内容）：\n"
                "- 不需要追问：PASS\n"
                "- 需要追问：CLARIFY:追问内容\n\n"
                "追问内容要求：\n"
                "1. 用\"您\"，语气友好，一次问齐缺失的关键信息，最多 2~3 个问题\n"
                "2. 参考历史对话，不要重复问用户已经提供过、或助手已经问过的信息\n"
                "3. 历史中形如\"CLARIFY:...\"的助手消息表示之前的一次追问，注意避免重复提问"),
                ("user", "历史对话记录：\n{history}\n\n用户当前问题：\n{input}"),
            ]
        )
        self.clarify_chain = self.clarify_prompt | self.chat_model | StrOutputParser()

        # ---- 环节三：生成链----
        # StrOutputParser只取最终答案；系统提示词再约束"只输出结论"作双保险
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是服装电商智能客服。只依据以下参考资料回答客户问题，资料中没有的内容不要编造。"
                "回答要简洁直接，只输出结论，不要输出思考过程和解释。\n\n参考资料：\n{context}"),
                ("system", "以下是用户的历史对话记录："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )
        self.answer_chain = self.prompt_template | self.chat_model | StrOutputParser()

        # 带历史记忆：每次问答自动写入对应会话的历史文件
        self.conversation_chain = RunnableWithMessageHistory(
            self.answer_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask(self, question: str, session_id: str = "default") -> dict:
        """对外主入口：改写 -> 检索 -> 兜底 / 生成
        返回 dict：answer=回答文本，sources=参考来源文件名列表
        """
        # 1. 查询改写
        if config.query_rewrite_on:
            rewritten = self.rewrite_chain.invoke({"input":question}).strip()
        else:
            rewritten = question

        # 2. 意图闸门：改写后、检索前，意图模糊则主动追问
        # 追问与否只由"意图"决定，不由"检索运气"决定——模糊问题即使侥幸
        # 命中低相关内容也不硬答，先问清需求
        if config.clarify_on and not self._clarify_reached(session_id, config.clarify_max_rounds):
            result = self.clarify_chain.invoke({
                "input": question,
                "history": self._history_to_text(get_history(session_id).messages),
            }).strip()
            if result.upper().startswith("CLARIFY:"):
                clarify_text = result[len("CLARIFY:"):].strip()
                if clarify_text:  # 防御：追问内容为空则按 PASS 处理
                    history = get_history(session_id)
                    history.add_user_message(question)
                    # 存带前缀原文：供 _clarify_reached 判环 + 历史文件可观测；
                    # 第二轮生成链看到 "CLARIFY:..." 也明确知道这是之前的追问
                    history.add_ai_message(result)
                    return {"answer": clarify_text, "sources": []}

        # 3. 阈值检索：低于阈值的文档被过滤，可能返回空列表
        docs = self.retriever.invoke(rewritten)

        # 4. 兜底：没有相关内容时不调用生成模型硬答
        if not docs:
            # 与追问分支同理：提前 return 不走生成链，历史包装器不会自动写入，
            # 手动落盘这一轮问答，保证无关问题的会话也完整可追溯
            history = get_history(session_id)
            history.add_user_message(question)
            history.add_ai_message(config.fallback_answer)
            return {"answer": config.fallback_answer, "sources": []}

        # 5. 组装参考资料。注意：原问题留给生成，改写词只用于检索
        context = self._format_document(docs)

        # 6. 带历史生成。历史包装器写的是链的输出
        # sources 只返回给前端展示，不进历史文件
        session_config = {"configurable": {"session_id": session_id}}
        answer = self.conversation_chain.invoke(
            {"input": question, "context": context}, session_config
        )

        # 7. 溯源信息：提取来源文件名，dict.fromkeys 去重且保持检索顺序
        sources = list(dict.fromkeys(doc.metadata.get("source", "") for doc in docs))
        return {"answer": answer, "sources": sources}

    @staticmethod
    def _history_to_text(messages):
        """把历史消息拼成 '用户：/助手：' 文本，供 clarify_chain 参考"""
        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}：{msg.content}")
        return "\n".join(lines) if lines else "（暂无历史对话）"

    def _clarify_reached(self, session_id, max_rounds):
        """连续追问轮数是否已达上限：防止追问死循环（轮数为 0 等价于关闭追问）"""
        if max_rounds <= 0:
            return True
        count = 0
        for msg in reversed(get_history(session_id).messages):
            if isinstance(msg, AIMessage):
                if msg.content.strip().startswith("CLARIFY:"):
                    count += 1
                    if count >= max_rounds:
                        return True
                else:
                    # 最近一次助手回复不是追问 → 上一轮已正常作答，连续追问中断
                    return False
        return False

    def _format_document(self, docs) -> str:
        """把检索到的多个文档片段格式化为参考资料文本"""
        formatted = ""
        for doc in docs:
            formatted += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
        return formatted


if __name__ == "__main__":
    service = RagService()
    res = service.ask("我体重180斤。尺码推荐", session_id="user_001")
    print(res["answer"])
    print("参考来源:",res["sources"])