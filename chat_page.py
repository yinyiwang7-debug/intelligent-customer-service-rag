"""
智能客服聊天页

由 app.py 统一入口调用 render_chat_page()，不要单独 streamlit run 本文件。

1. 浏览器会话隔离：session_id 用 uuid 随机生成，存 st.session_state
   （Streamlit 的 session_state 天然按浏览器会话隔离，每个标签页独立）
2. 侧边栏"清空历史记录"按钮
3. 回答溯源展示：答案下方展示参考来源文件
4. 异常友好提示：LLM 调用失败显示提示语，不再崩页面
"""
import secrets

import streamlit as st
from rag import RagService
from file_history_store import get_history


def render_chat_page():
    """渲染智能客服页：标题、会话管理侧边栏、消息列表、输入框"""
    # 标题
    st.title("智能客服")
    st.divider()

    # ---- 会话管理：session_id 挂在 URL 查询参数上 ----
    # 刷新/复制链接都不丢 id；多用户各用各的 URL 天然隔离
    # （生产等价的简化版：客户端携带标识，服务端按标识存取，无状态）
    params = st.query_params
    session_id = params.get("session_id")
    if not session_id:
        params["session_id"] = secrets.token_urlsafe(8)  # 首次访问生成短 token 并写进 URL
        st.rerun()                                        # 立即重跑，本次会话拿到新 id
    session_id = params["session_id"]

    # ---- 页面消息与服务初始化：刷新后从文件历史还原（欢迎语仅首次访问显示） ----
    if "message" not in st.session_state:
        history = get_history(session_id)
        if history.messages:
            # 文件历史里是 HumanMessage/AIMessage 对象，还原成页面消息列表
            st.session_state["message"] = [{"role": ("user" if m.type == "human" else "assistant"),
                                            "content": m.content}
                                           for m in history.messages]
        else:
            st.session_state["message"] = [{"role": "assistant", "content":
                "你好，请问有什么可以帮助您"}]

    if "rag" not in st.session_state:
        st.session_state["rag"] = RagService()

    # ---- 侧边栏：会话管理 ----
    with st.sidebar:
        st.subheader("会话管理")
        st.caption(f"会话ID：{session_id[:8]}...")
        if st.button("清空历史记录", key="clear_history_btn"):
            # 双清：历史文件 + 页面消息列表
            get_history(session_id).clear()
            st.session_state["message"] = [{"role": "assistant", "content":
                "历史已清空，请问有什么可以帮助您"}]
            st.rerun()

    # ---- 渲染历史消息 ----
    for message in st.session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])

    # ---- 用户输入 ----
    prompt = st.chat_input()

    if prompt:
        # 页面显示用户提问
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})

        # 异常友好提示：LLM 调用失败（如欠费、超时）不崩页面
        try:
            with st.spinner("AI思考中..."):
                result = st.session_state["rag"].ask(prompt,
                                                     session_id=session_id)

            answer = result["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state["message"].append({"role": "assistant",
                                                "content": answer})

            # 回答溯源展示：答案下方小字列出参考文件
            if result["sources"]:
                st.caption("参考资料来源：" + "、".join(result["sources"]))
        except Exception as e:
            # 只显示错误类型，不把堆栈甩给用户
            st.chat_message("assistant").error(f"服务暂时不可用，请稍后重试。（错误类型：{type(e).__name__}）")
