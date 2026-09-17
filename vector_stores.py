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
    # 自测入口：复用共享单例，离线模式与缓存目录统一由 config_data 决定，避免各处重复硬编码
    from embeddings import get_embeddings

    retriever = VectorStoreService(get_embeddings()).get_retriever()
    res = retriever.invoke("我的体重181斤，尺码推荐")
    print(res)
