"""
共享 Embedding 单例：整个应用只加载一次 bge-m3（约 2GB 内存 / 3-5 秒）

RagService 和 KnowledgeBaseService 各自 new 一个 HuggingFaceEmbeddings 的话，
模型会加载两遍、内存翻倍。这里用模块级缓存：谁先用到谁触发加载，
之后所有人复用同一个实例。原理与 st.session_state 缓存相同：缓存 + 复用。
"""
# 必须先导入 config_data：它在导入时按本地/云端设置 HF_HUB_OFFLINE，
# 该环境变量必须在 transformers（由 langchain_huggingface 引入）导入之前生效
import config_data as config
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # 构造参数统一读 config_data：换模型/改设备只动配置，不改本文件
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            cache_folder=config.embedding_cache_folder,
            model_kwargs={"device": config.embedding_device},
            encode_kwargs={"normalize_embeddings": config.embedding_normalize},
        )
    return _embeddings
