# -*- coding: utf-8 -*-
"""全量测试套件：8 组 64 项自动化 + 5 项手测（见 测试用例清单.md）

用法（在 智能客服助手 目录下）：
  ..\\.venv\\Scripts\\python.exe 测试脚本\\test_suite.py            # 全量（含真实 LLM，约 5-10 分钟）
  ..\\.venv\\Scripts\\python.exe 测试脚本\\test_suite.py A          # 只跑 A 组
  ..\\.venv\\Scripts\\python.exe 测试脚本\\test_suite.py A B C      # 跑多组
  ..\\.venv\\Scripts\\python.exe 测试脚本\\test_suite.py skip-llm   # 跳过真实 LLM 组（快测）

测试数据约定：文件名一律 bugtest_ 前缀，结束后自动清理，不污染知识库。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import hashlib
import re
from io import BytesIO

# ================= 基础设施 =================
RESULTS = []   # [(组, 名, 通过, 详情)]

def check(group, name, cond, extra=""):
    RESULTS.append((group, name, bool(cond), extra))
    mark = "PASS" if cond else "FAIL"
    print(f"  {mark} {name}" + (f" | {extra}" if extra else ""))

def make_pdf(text: str) -> bytes:
    """动态生成含 text 的最简合法 PDF"""
    stream = f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"%PDF-1.4",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj",
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream\nendobj",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    ]
    offsets, data = [], b""
    for i, o in enumerate(objs, 1):
        offsets.append(len(data)); data += o + b"\n"
    xref_pos = len(data)
    xref = f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    data += xref + f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return data

from docx import Document as DocxDoc
def make_docx(paras, table_data=None) -> bytes:
    doc = DocxDoc()
    for p in paras:
        doc.add_paragraph(p)
    if table_data:
        t = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
        for i, row in enumerate(table_data):
            for j, v in enumerate(row):
                t.cell(i, j).text = v
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()

# ================= A 组：数据管线 =================
def group_A():
    print("\n=== A 组：数据管线 ===")
    from data_pipeline import (_decode_bytes, _is_page_number, clean_text,
                               split_text, chunk_md5, process)

    # A1-A3 编码探测
    check("A", "A1 utf-8 解码", _decode_bytes("你好世界".encode("utf-8")) == "你好世界")
    check("A", "A2 gbk 解码", _decode_bytes("你好世界".encode("gbk")) == "你好世界")
    bad = "abc".encode() + b"\xff\xfe" + "def".encode()
    check("A", "A3 坏字节 ignore 兜底不抛异常", "def" in _decode_bytes(bad))

    # A4-A5 页码行判定
    check("A", "A4 纯数字判为页码", _is_page_number("12"))
    check("A", "A5 带字行不判为页码", not _is_page_number("第12页"))

    # A6-A8 清洗
    dirty = "标题\n12\n正文第一行\n页眉\n正文第二行\n页眉\n正文第三行\n页眉\n\n\n\n结尾"
    cleaned = clean_text(dirty)
    check("A", "A6 页码行删除", "12" not in cleaned.split("\n"))
    check("A", "A7 重复3次短行(页眉)删除", "页眉" not in cleaned.split("\n"))
    check("A", "A8 3+空行压缩", "\n\n\n" not in cleaned)

    # A9-A11 分割
    short = "短文本" * 10
    check("A", "A9 短文本不分割为1块", len(split_text(short, "txt")) == 1)
    long = "深蓝色直筒牛仔裤是百搭单品。" * 120   # 1680 字
    chunks = split_text(long, "txt")
    check("A", "A10 长文本分割>1块", len(chunks) >= 2)
    check("A", "A11 每块<=chunk_size", all(len(c) <= 1000 for c in chunks))
    check("A", "A12 相邻块有overlap重叠", chunks[0][-30:] in chunks[1] or chunks[1][:30] in chunks[0])

    # A13 md 两段式：标题保留 + 超长段再切
    md = "# 标题一\n\n" + "内容A。" * 200 + "\n\n# 标题二\n\n" + "内容B。" * 200
    md_chunks = split_text(md, "md")
    check("A", "A13 md按标题切成多块且标题保留在块内",
          len(md_chunks) >= 2 and any("标题一" in c for c in md_chunks) and any("标题二" in c for c in md_chunks))

    # A14 chunk 指纹归一化
    check("A", "A14 chunk_md5 忽略首尾空白", chunk_md5("  内容  ") == chunk_md5("内容"))

# ================= B 组：知识库 =================
def group_B():
    print("\n=== B 组：知识库 ===")
    from knowledge_base import (KnowledgeBaseService, read_md5_records,
                                find_md5_by_filename)
    kb = KnowledgeBaseService()
    # 清理可能的残留
    for f in [x["source"] for x in kb.list_files()]:
        if f.startswith("bugtest"):
            kb.delete_file(f)

    txt = "B组测试文本：" + "黑色西装裤百搭。" * 30
    md = "# 测试\n\n" + "B组md内容。" * 40
    docx = make_docx(["B组docx段落一", "B组docx段落二"],
                     [["面料", "特性"], ["棉", "透气"]])
    pdf = make_pdf("B Group PDF Text")

    # B1-B4 四种格式上传
    r = kb.uploader_file(txt.encode(), "bugtest_b1.txt")
    check("B", "B1 txt 上传成功", r.startswith("[成功]"), r[:40])
    r = kb.uploader_file(md.encode(), "bugtest_b2.md")
    check("B", "B2 md 上传成功", r.startswith("[成功]"), r[:40])
    r = kb.uploader_file(docx, "bugtest_b3.docx")
    check("B", "B3 docx 上传成功(含表格)", r.startswith("[成功]"), r[:40])
    r = kb.uploader_file(pdf, "bugtest_b4.pdf")
    check("B", "B4 pdf 上传成功", r.startswith("[成功]"), r[:40])

    # B5-B7 去重三态
    r = kb.uploader_file(txt.encode(), "bugtest_b1.txt")
    check("B", "B5 整文件重复跳过", r.startswith("[跳过]"), r[:40])
    r = kb.uploader_file((txt + "\n新增内容。").encode(), "bugtest_b1.txt")
    check("B", "B6 同名不同内容=替换", r.startswith("[成功]") and "已替换旧版本" in r, r[:40])
    # chunk 级：共享前缀超 1000 字使块边界落在共享区
    prefix = "深蓝色直筒牛仔裤是百搭单品。" * 72
    r = kb.uploader_file((prefix + "A独有。" * 20).encode(), "bugtest_ca.txt")
    r = kb.uploader_file((prefix + "B独有。" * 20).encode(), "bugtest_cb.txt")
    check("B", "B7 chunk级去重跳过1块", r.startswith("[成功]") and "跳过重复 1 块" in r, r[:50])

    # B8-B11 异常输入
    r = kb.uploader_file(b"x", "bugtest_noext")
    check("B", "B8 无扩展名拒绝", r.startswith("[失败]"))
    r = kb.uploader_file(b"x", "bugtest.exe")
    check("B", "B9 非法扩展名拒绝", r.startswith("[失败]"))
    r = kb.uploader_file(b"", "bugtest_empty.txt")
    check("B", "B10 空文件拒绝", r.startswith("[失败]"))
    r = kb.uploader_file(make_pdf(""), "bugtest_scan.pdf")  # 无文本 PDF
    check("B", "B11 扫描版PDF拒绝", r.startswith("[失败]"), r[:50])

    # B12 文件名路径穿越安全
    r = kb.uploader_file("穿越测试内容".encode(), "../../etc/bugtest_evil.txt")
    check("B", "B12 路径穿越被截断为文件名", r.startswith("[成功]") and "bugtest_evil.txt" in r, r[:50])
    kb.delete_file("bugtest_evil.txt")

    # B13-B15 删除
    kb.uploader_file("临时删除测试".encode(), "bugtest_del.txt")
    r = kb.delete_file("bugtest_del.txt")
    check("B", "B13 删除存在文件", r.startswith("[成功]"))
    check("B", "B14 删除后md5记录同步清理",
          find_md5_by_filename(read_md5_records(), "bugtest_del.txt") is None)
    r = kb.delete_file("bugtest_del.txt")
    check("B", "B15 删除不存在文件返回失败", r.startswith("[失败]"))
    r = kb.uploader_file("临时删除测试".encode(), "bugtest_del.txt")
    check("B", "B16 删后重传=首次上传", r.startswith("[成功]") and "已上传" in r and "替换" not in r)

    # 清理本组测试文件
    for f in [x["source"] for x in kb.list_files()]:
        if f.startswith("bugtest"):
            kb.delete_file(f)

# ================= C 组：检索 =================
def group_C():
    print("\n=== C 组：检索 ===")
    from vector_stores import VectorStoreService
    from embeddings import get_embeddings
    vs = VectorStoreService(get_embeddings())
    retriever = vs.get_retriever()

    # C1 相关文档被召回
    docs = retriever.invoke("衣服怎么洗涤")
    check("C", "C1 相关问题召回文档", len(docs) >= 1,
          f"召回{len(docs)}个: {[d.metadata.get('source') for d in docs]}")

    # C2 召回数量 <= k
    check("C", "C2 召回数<=k(4)", len(docs) <= 4)

    # C3 无关问题返回空
    docs2 = retriever.invoke("量子力学薛定谔方程推导过程")
    check("C", "C3 无关问题返回空列表", len(docs2) == 0)

    # C4 阈值过滤生效：retriever 返回的文档数 == 分数>=阈值的文档数
    docs3 = retriever.invoke("尺码推荐")
    all_docs_scores = vs.vector_store.similarity_search_with_relevance_scores("尺码推荐", k=4)
    kept = [d for d, s in all_docs_scores if s >= 0.30]   # 与 config 阈值一致
    check("C", "C4 阈值过滤生效(返回数==达标数)", len(docs3) == len(kept),
          f"返回{len(docs3)}个, 达标{len(kept)}个")

# ================= D 组：历史 =================
def group_D():
    print("\n=== D 组：历史 ===")
    from file_history_store import get_history
    from langchain_core.messages import HumanMessage, AIMessage
    import uuid as _uuid

    # D1 空会话
    h = get_history("bugtest_d_empty")
    h.clear()
    check("D", "D1 空会话返回[]", h.messages == [])

    # D2 读写往返
    h2 = get_history("bugtest_d_rw")
    h2.clear()
    h2.add_messages([HumanMessage(content="问题X"), AIMessage(content="回答Y")])
    msgs = get_history("bugtest_d_rw").messages
    check("D", "D2 写读往返一致", len(msgs) == 2 and msgs[0].content == "问题X" and msgs[1].content == "回答Y")

    # D3 裁剪保留最新
    h3 = get_history("bugtest_d_cut")
    h3.clear()
    for i in range(12):
        h3.add_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])
    msgs = h3.messages
    check("D", "D3 12条裁剪为8条", len(msgs) == 8)
    check("D", "D4 裁剪保留最新消息", msgs[-1].content == "a11")

    # D5 多会话隔离
    check("D", "D5 不同session互相隔离",
          get_history("bugtest_d_empty").messages == [] and len(get_history("bugtest_d_cut").messages) == 8)

    for sid in ["bugtest_d_empty", "bugtest_d_rw", "bugtest_d_cut"]:
        get_history(sid).clear()

# ================= E 组：问答（真实 LLM） =================
def group_E():
    print("\n=== E 组：问答（真实 LLM，较慢）===")
    from rag import RagService
    from file_history_store import get_history
    rag = RagService()
    sess = "bugtest_e_sess"
    get_history(sess).clear()

    # E1 精确问题有溯源
    r = rag.ask("衣服应该怎么洗涤养护？", sess)
    check("E", "E1 精确问题有溯源", len(r.get("sources", [])) >= 1, r.get("answer", "")[:30])

    # E2 信息不全触发追问（页面返回的追问文本已去掉 CLARIFY: 前缀，前缀只在落盘时保留）
    r = rag.ask("我体重180斤，穿什么尺码？", sess)
    ans = r.get("answer", "")
    check("E", "E2 信息不全触发追问", r.get("sources") == [] and "请问" in ans, ans[:40])

    # E3 追问轮已落盘
    contents = [m.content for m in get_history(sess).messages]
    check("E", "E3 追问带前缀落盘", any("CLARIFY:" in c for c in contents), f"{len(contents)}条")

    # E4 信息补充后完整回答
    r = rag.ask("身高180体重75公斤的男生夏天穿什么码T恤？", sess)
    check("E", "E4 补充信息后正常回答", r.get("sources") != [], r.get("answer", "")[:30])

    # E5 判环：clarify_max_rounds=1，本会话已有1次追问，再问信息不全应不再追问
    r = rag.ask("我体重160斤穿多大码？", sess)
    check("E", "E5 追问达上限后不再追问", "CLARIFY:" not in r.get("answer", ""), r.get("answer", "")[:40])

    # E6 裁剪上限下的滚动写入：历史已到 8 条上限，新写入应挤掉最旧、保持 8 条且内容更新
    n_before = len(get_history(sess).messages)
    r = rag.ask("洗涤的时候要注意什么？", sess)
    msgs = get_history(sess).messages
    n_after = len(msgs)
    check("E", "E6 达上限后滚动写入(保持8条且最新为AI)",
          n_before == 8 and n_after == 8 and msgs[-1].type == "ai" and bool(msgs[-1].content),
          f"{n_before}->{n_after}, 最后一条类型={msgs[-1].type if msgs else '无'}")

    # E7 无关问题走兜底
    sess2 = "bugtest_e_sess2"
    get_history(sess2).clear()
    r = rag.ask("今天天气怎么样？", sess2)
    check("E", "E7 无关问题走兜底", "人工客服" in r.get("answer", "") or "没有" in r.get("answer", ""), r.get("answer", "")[:40])

    # E8 兜底轮也手动落盘
    contents2 = [m.content for m in get_history(sess2).messages]
    check("E", "E8 兜底轮手动落盘", len(contents2) >= 2, f"{len(contents2)}条")

    get_history(sess).clear()
    get_history(sess2).clear()

# ================= F 组：API =================
def group_F():
    print("\n=== F 组：API ===")
    from fastapi.testclient import TestClient
    import api

    with TestClient(api.app) as client:
        # F1-F3 上传/列表/删除
        content = "API测试内容：" + "黑色皮鞋配深色西装裤最正式。" * 30
        r = client.post("/upload", files={"file": ("bugtest_f1.txt", content.encode(), "text/plain")})
        check("F", "F1 POST /upload", r.status_code == 200 and "已上传" in r.json()["message"])
        r = client.get("/files")
        names = [f["source"] for f in r.json()]
        check("F", "F2 GET /files 含新文件", r.status_code == 200 and "bugtest_f1.txt" in names)
        r = client.delete("/files/bugtest_f1.txt")
        check("F", "F3 DELETE /files", r.status_code == 200 and "已删除" in r.json()["message"])

        # F4-F5 聊天与历史
        r = client.post("/chat", json={"question": "衣服怎么洗？", "session_id": "bugtest_f_sess"})
        j = r.json()
        check("F", "F4 POST /chat", r.status_code == 200 and bool(j.get("answer")))
        r = client.get("/history/bugtest_f_sess")
        check("F", "F5 GET /history", r.status_code == 200 and len(r.json()) >= 1)
        r = client.delete("/history/bugtest_f_sess")
        check("F", "F6 DELETE /history", r.status_code == 200)

        # F7 缺字段 422
        r = client.post("/chat", json={"session_id": "x"})
        check("F", "F7 chat 缺 question 返回422", r.status_code == 422)

        # F8 非法文件名（路径穿越）
        r = client.delete("/files/..%2F..%2Fevil.txt")
        check("F", "F8 删除非法文件名不崩溃", r.status_code in (200, 404), f"status={r.status_code}")

        # F9 不存在的会话历史返回 []
        r = client.get("/history/bugtest_not_exist_xyz")
        check("F", "F9 不存在会话返回[]", r.status_code == 200 and r.json() == [])

# ================= G 组：页面（AppTest） =================
def group_G():
    print("\n=== G 组：页面 ===")
    from streamlit.testing.v1 import AppTest
    from file_history_store import get_history
    from langchain_core.messages import HumanMessage, AIMessage

    # AppTest.from_file 相对路径按"调用脚本所在目录"解析，这里用绝对路径
    APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

    # G1 导航默认页 + URL 持久化
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    check("G", "G1 首次默认智能客服页", at.radio[0].value == "智能客服")
    at.radio[0].set_value("知识库管理")
    at.run()
    check("G", "G2 切页写入URL参数", "知识库管理" in at.query_params.get("page", []))
    at2 = AppTest.from_file(APP_PATH, default_timeout=180)
    at2.query_params["page"] = "知识库管理"
    at2.run()
    check("G", "G3 刷新停在原页", at2.radio[0].value == "知识库管理")

    # G4-G5 上传幂等（file_id 指纹）
    at3 = AppTest.from_file(APP_PATH, default_timeout=180)
    at3.query_params["page"] = "知识库管理"
    at3.run()
    kb = at3.session_state["kb_service"]
    for f in [x["source"] for x in kb.list_files()]:
        if f.startswith("bugtest"):
            kb.delete_file(f)
    at3.run()
    at3.file_uploader[0].set_value(("bugtest_g1.txt", "G组幂等测试内容".encode(), "text/plain"))
    at3.run()
    check("G", "G4 上传后绿色提示", len(at3.success) >= 1)
    at3.run()
    cnt = [x["source"] for x in kb.list_files()].count("bugtest_g1.txt")
    check("G", "G5 rerun不重复上传", cnt == 1)

    # G6 删除后重传同文件正常
    btn = next(b for b in at3.button if b.key == "del_bugtest_g1.txt")
    btn.click()
    at3.run()
    at3.file_uploader[0].set_value(("bugtest_g1.txt", "G组幂等测试内容".encode(), "text/plain"))
    at3.run()
    check("G", "G6 删后重传正常入库", "bugtest_g1.txt" in [x["source"] for x in kb.list_files()])

    # G7-G8 聊天页历史还原
    h = get_history("apptest_gh")
    h.clear()
    h.add_messages([HumanMessage(content="G组历史问题"), AIMessage(content="G组历史回答")])
    at4 = AppTest.from_file(APP_PATH, default_timeout=180)
    at4.query_params["session_id"] = "apptest_gh"
    at4.run()
    md_texts = [getattr(x, "value", "") for x in at4.markdown]
    check("G", "G7 刷新后历史消息还原显示", any("G组历史问题" in t for t in md_texts), md_texts)
    check("G", "G8 欢迎语不重复出现", "你好，请问有什么可以帮助您" not in md_texts)

    kb.delete_file("bugtest_g1.txt")
    h.clear()

# ================= H 组：一致性与资源 =================
def group_H():
    print("\n=== H 组：一致性与资源 ===")
    from knowledge_base import KnowledgeBaseService, read_md5_records
    from embeddings import get_embeddings

    # H1 单例
    check("H", "H1 embeddings 单例", get_embeddings() is get_embeddings())

    # H2 幽灵记录检测：md5 记录里的文件名必须都存在于向量库
    kb = KnowledgeBaseService()
    files = {x["source"] for x in kb.list_files()}
    records = read_md5_records()
    ghosts = [(md5, name) for md5, name in records.items() if name and name not in files]
    check("H", "H2 无幽灵md5记录(有文件名但在库里不存在)", len(ghosts) == 0, str(ghosts)[:60])

    # H3 md5 记录数 == 知识库文件数
    check("H", "H3 md5记录数与文件数一致", len(records) == len(files),
          f"记录{len(records)} vs 文件{len(files)}")

    # H4 历史裁剪防膨胀
    from file_history_store import get_history
    h = get_history("bugtest_h4")
    h.clear()
    check("H", "H4 清空后文件为空列表", h.messages == [])

    # H5 知识库文件与 data/ 原始文件能对应
    data_dir = Path("data")
    data_files = {f.name for f in data_dir.iterdir() if f.is_file()}
    kb_files = {x["source"] for x in kb.list_files()}
    check("H", "H5 知识库文件均在 data/ 有原件", kb_files.issubset(data_files),
          f"库里{len(kb_files)}个, data/有{len(data_files)}个")

# ================= 主入口 =================
if __name__ == "__main__":
    args = sys.argv[1:]
    groups_all = {"A": group_A, "B": group_B, "C": group_C, "D": group_D,
                  "E": group_E, "F": group_F, "G": group_G, "H": group_H}
    if "skip-llm" in args:
        args.remove("skip-llm")
        groups_all.pop("E", None)
        print("[skip-llm] 跳过真实 LLM 组")
    if not args:
        args = list(groups_all.keys())
    for g in args:
        g = g.upper()
        if g not in groups_all:
            print(f"未知组: {g}，可选: {list(groups_all.keys())} / skip-llm")
            continue
        groups_all[g]()

    total = len(RESULTS)
    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    print(f"\n===== 汇总: {passed}/{total} 通过 =====")
    for g, name, ok, extra in RESULTS:
        if not ok:
            print(f"  FAIL [{g}] {name} | {extra}")
