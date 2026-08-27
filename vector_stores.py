from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )


    def get_retriever(self):
        """ 返回向量检索器，方便加入chain"""
        # 阈值检索模式：只返回相似度 >= score_threshold的文档，无关内容被过滤（可能返回空列表）
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.retrieve_k,
                "score_threshold": config.similarity_threshold,
            }
        )

if __name__ == '__main__':
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from langchain_huggingface import HuggingFaceEmbeddings

    retriever = VectorStoreService(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            cache_folder="D:/huggingface_cache",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    ).get_retriever()

    res = retriever.invoke("我的体重181斤，尺码推荐")
    print(res)
