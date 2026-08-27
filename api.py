"""
FastAPI 接口层（与 Streamlit 页面并存，业务逻辑完全复用现有 service）

启动：cd RAG项目 && ..\\.venv\\Scripts\\python.exe -m uvicorn api:app --port 8000
接口文档（自动生成）：http://127.0.0.1:8000/docs

六个接口：
  POST   /upload                 上传文件到知识库（multipart）
  GET    /files                  知识库文件列表
  DELETE /files/{filename}       删除知识库文件
  POST   /chat                   问答（question + session_id）
  GET    /history/{session_id}   查看会话历史
  DELETE /history/{session_id}   清空会话历史
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from dotenv import load_dotenv
# .env 在项目根目录（RAG项目 的上一级），绝对路径加载，不受启动目录影响
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from knowledge_base import KnowledgeBaseService
from rag import RagService
from file_history_store import get_history


# ---- 服务单例：启动时加载一次（bge-m3 约 3-5 秒），之后所有请求共享 ----
# 与 Streamlit 的 session_state 缓存同一思想，只是生命周期换成"服务进程"
_kb_service = None
_rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：服务启动时加载模型，关闭时回收"""
    global _kb_service, _rag_service
    print("加载知识库服务...")
    _kb_service = KnowledgeBaseService()
    print("加载问答服务...")
    _rag_service = RagService()   # 与 kb_service 共享 embeddings 单例，模型只加载一次
    print("服务就绪，接口文档见 http://127.0.0.1:8000/docs")
    yield
    # 退出时无需特殊清理（Chroma 无连接池）


app = FastAPI(title="RAG 智能客服 API", lifespan=lifespan)

# 预留：以后浏览器前端跨域调用时放行（本地开发全开）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求/响应模型（pydantic：自动校验参数 + 自动生成接口文档）----
class ChatRequest(BaseModel):
    question: str                          # 必填：用户问题
    session_id: Optional[str] = "default"  # 选填：会话 ID，决定读写哪个历史文件


class ChatResponse(BaseModel):
    answer: str                            # 回答文本
    sources: list[str]                     # 参考来源文件名列表


# ================= 知识库接口 =================

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """上传文件到知识库（txt / pdf / docx / md）"""
    # 路由声明为普通 def（同步）：FastAPI 自动丢线程池执行，
    # 文件解析 + embedding 是耗时 CPU 操作，不能阻塞事件循环
    data = file.file.read()                # 读底层文件对象的原始字节
    result = _kb_service.uploader_file(data, file.filename or "")
    if result.startswith("[失败]"):
        raise HTTPException(status_code=400, detail=result)
    return {"message": result}


@app.get("/files")
def list_files():
    """知识库文件列表（块数 / 上传时间，倒序）"""
    return _kb_service.list_files()


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """删除知识库中某文件的全部向量 + 清理去重指纹"""
    result = _kb_service.delete_file(filename)
    if result.startswith("[失败]"):
        raise HTTPException(status_code=404, detail=result)
    return {"message": result}


# ================= 问答接口 =================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """问答主接口：改写 -> 阈值检索 -> 兜底/生成，返回 answer + 溯源 sources"""
    try:
        return _rag_service.ask(req.question, session_id=req.session_id)
    except Exception as e:
        # LLM 调用失败（欠费/超时）时给调用方明确的 502，而不是裸 500 堆栈
        raise HTTPException(status_code=502, detail=f"LLM 服务不可用：{type(e).__name__}")


# ================= 会话历史接口 =================

@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    """查看某会话的历史消息（role + content 列表）"""
    history = get_history(session_id)
    return [{"role": m.type, "content": m.content} for m in history.messages]


@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str):
    """清空某会话的历史消息"""
    get_history(session_id).clear()
    return {"message": f"会话 {session_id} 历史已清空"}
