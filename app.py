"""
RAG 项目统一入口：访问口令门禁 + 知识库自举 + 侧边栏导航

原理：Streamlit 每次交互都会从头重跑脚本。
用 if 分支只渲染"选中的页面"，未选中页面的代码不会执行
-> 它的 service 不会被构造 -> 模型按需加载、全应用只加载一次。
"""
import os

import streamlit as st

# 必须早于 chat_page / kb_page 导入：config_data 在导入时决定 HF 离线模式与缓存目录
import config_data as config
from chat_page import render_chat_page
from kb_page import render_kb_page

# 必须是脚本第一个 st 命令：控制浏览器标签页标题
st.set_page_config(page_title="RAG 学习项目")


# ---- 访问口令门禁 ----
# 公网部署后链接谁都能打开，不加门禁陌生人的每一次提问都在消耗你的 DeepSeek 额度。
# 本地未配置 APP_PASSWORD 时自动放行，不影响日常开发。
def _access_ok() -> bool:
    expect = config.get_secret("APP_PASSWORD")
    if not expect:
        return True                                     # 未配置口令 = 本地开发模式，直接放行
    if st.session_state.get("_auth_ok"):
        return True                                     # 本次会话已通过验证，不再重复拦
    pwd = st.text_input("访问口令", type="password", placeholder="请输入访问口令")
    if pwd == expect:
        st.session_state["_auth_ok"] = True
        st.rerun()                                      # 验证通过后重跑，清掉输入框
    elif pwd:
        st.error("口令错误")
    return False


if not _access_ok():
    st.stop()


# ---- 知识库自举 ----
# 云端 chroma_db 被 .gitignore 排除，克隆下来是空的，不灌数据问答会全部走兜底文案。
@st.cache_resource(show_spinner="首次启动：正在构建知识库（加载 bge-m3 模型，约 1-3 分钟）...")
def _seed_knowledge_base():
    """把 data/ 下的种子文档灌进知识库；cache_resource 保证整个进程只执行一次"""
    from pathlib import Path

    from knowledge_base import KnowledgeBaseService

    service = KnowledgeBaseService()
    data_dir = Path(__file__).resolve().parent / "data"
    return [service.uploader_file(f.read_bytes(), f.name)
            for f in sorted(data_dir.glob("*.txt"))]


# 只做目录级判断（不加载模型）：本地已有 chroma_db -> 非空 -> 零开销跳过
if not os.path.isdir(config.persist_directory) or not os.listdir(config.persist_directory):
    _seed_knowledge_base()


# ---- 侧边栏导航：选中页挂 URL 参数，刷新后停在当前页 ----
params = st.query_params
page = params.get("page")
if page not in ("智能客服", "知识库管理"):
    params["page"] = "智能客服"   # 首次访问写默认页
    st.rerun()
page = params["page"]

with st.sidebar:
    st.subheader("页面导航")
    selected = st.radio("选择页面", ["智能客服", "知识库管理"],
                        index=0 if page == "智能客服" else 1, key="nav_page")
    if selected != page:
        params["page"] = selected   # 用户切换页面 -> 同步写回 URL
        st.rerun()

# ---- 按选中页面分发 ----
if page == "智能客服":
    render_chat_page()
else:
    render_kb_page()
