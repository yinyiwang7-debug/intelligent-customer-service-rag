"""
RAG 项目统一入口：侧边栏导航 + 按需渲染页面

原理：Streamlit 每次交互都会从头重跑脚本。
用 if 分支只渲染"选中的页面"，未选中页面的代码不会执行
-> 它的 service 不会被构造 -> 模型按需加载、全应用只加载一次。
"""
import streamlit as st
from chat_page import render_chat_page
from kb_page import render_kb_page

# 必须是脚本第一个 st 命令：控制浏览器标签页标题
st.set_page_config(page_title="RAG 学习项目")

# ---- 侧边栏导航：选中页挂 URL 参数，刷新后停在当前页 ----
# （与 chat_page 的 session_id 同一思路：URL 即状态，刷新不丢）
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

# ---- 按选中页面分发（未选中页面的代码不会执行）----
if page == "智能客服":
    render_chat_page()
else:
    render_kb_page()
