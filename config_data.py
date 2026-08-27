
md5_path = "./md5.text"                  # md5存储路径


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# Embedding（bge-m3）：构造参数
embedding_model_name = "BAAI/bge-m3"
embedding_cache_folder = "D:/huggingface_cache"
embedding_device = "cpu"
embedding_normalize = True      # 归一化后检索分数 = 余弦相似度

# spliter
chunk_size = 1000               # 分块最大长度
chunk_overlap = 100             # 允许块间最大重叠字数
separators = ["\n\n","\n",".","?","!","。","？","！"," ",""]       # 分割符列表（从大到小排列，保证先按段落再按句子切）
max_spliter_char_number = 1000          # 文本分隔的阈值

retrieve_k = 4                  # 检索返回的最大候选文档数量
similarity_threshold = 0.30     # 相似度阈值：分数低于此值的文档视为不相关


# 自测时设定的用户id
session_config = {
    "configurable": {
        "session_id": "user_001"
    }
}


# 支持的文档格式
support_file_types = ["txt", "pdf", "docx", "md"]

# 文本清洗开关
clean_header_footer = True  # 删除跨页重复出现的短行（页眉/页脚）
clean_page_number = True  # 删除整行为纯数字的行（页码）
compress_blank = True  # 连续空行压缩

splitter_config = {
    "txt": {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    },
    "pdf": {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    },
    "docx": {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    },

    # markdown 两段式：先按标题切结构，再按长度切
    "md": {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "headers_to_split_on": [
            ("#", "标题1"),
            ("##", "标题2"),
            ("###", "标题3"),
        ],
        "separators": separators,
    },
}

#查询改写开关：True = 先改写再检索，False = 原问题直接检索
query_rewrite_on = True
# 检索不到相关内容时的兜底回复
fallback_answer = "抱歉，知识库中暂时没有与您问题相关的信息，建议转接人工客服咨询。"

# 意图模糊主动追问开关：True = 用户需求不明确时先反问收集信息，False = 跳过追问
clarify_on = True
# 连续追问最多轮数：防止无限追问死循环
clarify_max_rounds = 1


max_history_messages = 8  # 历史消息条数上限：超出后只保留最近 N 条（防prompt 上下文无限膨胀）
