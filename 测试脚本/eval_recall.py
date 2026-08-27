"""
recall@k 评测脚本（检索效果回归基线）

评测链路与 rag.py 线上检索路径完全一致：
    用户问题 -> 查询改写链（LLM，query_rewrite_on=True 时） -> 阈值检索
                                            （score_threshold=0.30, k=4）
判定：每题标准文档是否出现在返回的前 k 个文档的 source 里。
     recall@k = 前 k 个里命中标准文档的题数 / 总题数

输出：
1. 每题明细（原问题 / 改写词 / top4 文档与相似度分数 / 是否命中）
2. recall@1 / recall@2 / recall@4 汇总
3. 失败题分析（目标文档在全量排序中的真实排位与分数，区分"阈值卡掉"和"排名不够"）
4. 结果写入 eval_baseline.txt，作为基线供以后调参回归对比

用法：
    cd 智能客服助手 && ../.venv/Scripts/python.exe 测试脚本/eval_recall.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from rag import RagService
from vector_stores import VectorStoreService
from embeddings import get_embeddings
import config_data as config
from eval_set import QA_PAIRS


def main():
    rag = RagService()
    # 失败题分析用句柄：similarity_search_with_relevance_scores
    # retriever.invoke 阈值检索的底层实现，参数保持一致
    vs = VectorStoreService(get_embeddings())

    total = len(QA_PAIRS)
    hits = {1: 0, 2: 0, 4: 0}
    failed = []  # (编号, 原问题, 改写词, top4源列表, 目标文档排位与分数)

    print(f"\n===== recall 评测：共 {total} 题（阈值 {config.similarity_threshold}, "
          f"k={config.retrieve_k}）=====\n")

    for i, (question, target) in enumerate(QA_PAIRS, 1):
        # 1. 查询改写：与线上一致
        rewritten = None
        if config.query_rewrite_on:
            try:
                rewritten = rag.rewrite_chain.invoke({"input": question}).strip()
            except Exception as e:
                print(f"[警告] Q{i} 改写失败（{type(e).__name__}），用原问题检索")
        if not rewritten:
            rewritten = question

        # 2. 阈值检索：与线上完全相同的调用
        docs = rag.retriever.invoke(rewritten)
        sources = [d.metadata.get("source", "") for d in docs]

        # 3. 判定：标准文档是否出现在前 k 个里
        row_hits = {k: target in sources[:k] for k in (1, 2, 4)}
        for k, ok in row_hits.items():
            hits[k] += 1 if ok else 0

        # 4. 每题明细打印
        mark = "PASS" if row_hits[4] else "FAIL"
        detail = " | ".join(f"{s}({sc:.2f})" for d, sc in
                            vs.vector_store.similarity_search_with_relevance_scores(
                                rewritten, k=config.retrieve_k,
                                score_threshold=config.similarity_threshold)
                            for s in [d.metadata.get("source", "")])
        print(f"[{mark}] Q{i}: {question}")
        print(f"改写: {rewritten}")
        print(f"top{config.retrieve_k}: {detail or '（阈值过滤后为空）'}")

        if not row_hits[4]:
            # 失败题分析：不带阈值取全量排序
            all_docs = vs.vector_store.similarity_search_with_relevance_scores(
                rewritten, k=20)
            pos = next((idx + 1 for idx, (d, sc) in enumerate(all_docs)
                        if d.metadata.get("source", "") == target), None)
            target_score = next((sc for d, sc in all_docs
                                 if d.metadata.get("source", "") == target), None)
            failed.append((i, question, rewritten, sources, pos, target_score))

    # 5. 汇总
    print(f"\n===== 汇总（阈值 {config.similarity_threshold}, k={config.retrieve_k}）=====")
    for k in (1, 2, 4):
        print(f"  recall@{k} = {hits[k]}/{total} = {hits[k] / total:.3f}")

    # 6. 失败题根因分析
    if failed:
        print(f"\n===== 失败题分析（{len(failed)} 题未命中 recall@4）=====")
        for i, q, rw, srcs, pos, score in failed:
            if score is not None and score < config.similarity_threshold:
                # 目标分数低于阈值：无论排第几都会被过滤 -> 根因是阈值过高
                cause = (f"阈值 {config.similarity_threshold} 把目标文档过滤了"
                         f"（目标排第 {pos} 位，分数 {score:.2f} < {config.similarity_threshold}）")
            elif pos is not None:
                # 分数达标但没进前 k -> 根因是排名被其他文档挤下
                cause = f"目标分数达标（{score:.2f}）但排第 {pos} 位，未进前 {config.retrieve_k}"
            else:
                # 全量前 20 都没有 -> 根因是改写词/嵌入与目标内容偏离
                cause = "目标文档未进入全量前 20：改写词/嵌入偏离目标内容"
            print(f"Q{i}「{q}」 改写「{rw}」 -> 实际返回 {srcs}；{cause}")
    else:
        print("\n===== 失败题分析：无，全部命中 =====")

    # 7. 基线写入文件（调参后重跑可对比；重复运行会覆盖为最新结果）
    from datetime import datetime
    report = Path(__file__).resolve().parent / "eval_baseline.txt"
    lines = [
        f"评测基线（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        f"配置: score_threshold={config.similarity_threshold}, "
        f"retrieve_k={config.retrieve_k}, query_rewrite_on={config.query_rewrite_on}",
        f"总题数: {total}",
    ]
    lines += [f"recall@{k} = {hits[k]}/{total} = {hits[k] / total:.3f}" for k in (1, 2, 4)]
    if failed:
        lines.append(f"失败题: {len(failed)} 题 -> {[f'Q{i}' for i, *_ in failed]}")
    else:
        lines.append("失败题: 无")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n基线已写入: {report.name}")


if __name__ == "__main__":
    main()
