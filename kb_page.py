"""
知识库管理页

由 app.py 统一入口调用 render_kb_page()，不要单独 streamlit run 本文件。

基于Streamlit完成WEB网页上传服务 + 知识库文件管理（列表/删除）
Streamlit: 当WEB页面元素发生变化，则代码重新执行一遍
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import config_data as config


def render_kb_page():
    """渲染知识库管理页：上传区 + 文件列表区"""
    st.title("知识库更新服务")

    # session_state就是一个字典：模型服务只初始化一次（页面重跑不会重复加载模型）
    if "kb_service" not in st.session_state:
        st.session_state["kb_service"] = KnowledgeBaseService()

    # ================= 上传区 =================
    st.divider()
    st.subheader("上传文件")

    uploader_file = st.file_uploader(
        "请上传文档文件（txt / pdf / docx / md）",
        type=config.support_file_types,
        accept_multiple_files=False,        # 表示仅接受一个文件的上传
        key="kb_uploader",                  # 显式key：切页后未提交的文件选择不会丢失
    )

    if uploader_file is not None:
        data = uploader_file.getvalue()
        fp = uploader_file.file_id
        if fp != st.session_state.get("last_uploaded_fp"):
            with st.spinner("载入知识库中..."):
                result = st.session_state["kb_service"].uploader_file(data, uploader_file.name)
            # 三色长驻提示：成功绿 / 跳过黄 / 失败红
            if result.startswith("[成功]"):
                st.success(result)
            elif result.startswith("[跳过]"):
                st.warning(result)
            else:
                st.error(result)
            st.session_state["last_uploaded_fp"] = fp

    # ================= 文件列表区 =================
    st.divider()
    st.subheader("知识库文件列表")

    files = st.session_state["kb_service"].list_files()
    if not files:
        st.info("知识库为空，请先上传文件")
    else:
        for info in files:
            # 一行四列：文件名 | 块数 | 上传时间 | 删除按钮
            col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
            col1.write(info["source"])
            col2.write(f"{info['chunk_count']} 块")
            col3.write(info["create_time"] or "未知")
            # 删除按钮的 key 必须全局唯一，用文件名拼前缀
            if col4.button("删除", key=f"del_{info['source']}"):
                result = st.session_state["kb_service"].delete_file(info["source"])
                st.toast(result)     # toast 会在 st.rerun 之后仍短暂显示
                st.rerun()           # 重新执行脚本，刷新列表
