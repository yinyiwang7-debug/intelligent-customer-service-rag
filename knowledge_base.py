import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from dotenv import load_dotenv
# .env 位于项目根目录，用绝对路径加载，不受启动目录影响
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import hashlib
from datetime import datetime

from embeddings import get_embeddings
from langchain_chroma import Chroma

import config_data as config
from data_pipeline import process, chunk_md5


# ========== 文件级 md5 记录：{md5: 文件名}，持久化在 config.md5_path ==========

def read_md5_records() -> dict:
  """读取全部文件级去重记录；新格式 md5|文件名，兼容旧格式纯md5（文件名记为空）"""
  records = {}
  if not os.path.exists(config.md5_path):
      return records
  with open(config.md5_path, "r", encoding="utf-8") as f:
      for line in f:
          line = line.strip()
          if not line:
              continue
          parts = line.split("|", 1)
          records[parts[0]] = parts[1] if len(parts) > 1 else ""
  return records


def save_md5_record(md5_str: str, filename: str) -> None:
  """追加一条记录"""
  with open(config.md5_path, "a", encoding="utf-8") as f:
      f.write(f"{md5_str}|{filename}\n")


def remove_md5_record(md5_str: str) -> None:
  """删除一条记录（同名替换时，清掉旧内容的指纹）"""
  records = read_md5_records()
  records.pop(md5_str, None)
  with open(config.md5_path, "w", encoding="utf-8") as f:
      for md5, name in records.items():
          f.write(f"{md5}|{name}\n")


def find_md5_by_filename(records: dict, filename: str):
  """反查：某文件名当前对应的内容指纹（无则返回 None）"""
  for md5, name in records.items():
      if name == filename:
          return md5
  return None


class KnowledgeBaseService(object):
    def __init__(self):
      os.makedirs(config.persist_directory, exist_ok=True)
      self.chroma = Chroma(
          collection_name=config.collection_name,
          embedding_function=get_embeddings(),   # 与 rag.py 共享同一个模型实例
          persist_directory=config.persist_directory,
      )

    def uploader_file(self, data: bytes, filename: str) -> str:
      """上传入口：文件字节 + 文件名。返回处理结果提示文字"""
      # 文件名安全：去掉路径部分，只保留文件名本身
      filename = filename.replace("\\", "/").split("/")[-1]

      # 按扩展名路由解析
      if "." not in filename:
          return "[失败]文件名缺少扩展名，无法判断格式"
      file_type = filename.rsplit(".", 1)[1].lower()
      if file_type not in config.support_file_types:
          return f"[失败]不支持的格式 .{file_type}，仅支持：{config.support_file_types}"

      # ---- 文件级去重判定 ----
      file_md5 = hashlib.md5(data).hexdigest()
      records = read_md5_records()

      if file_md5 in records:
          old_name = records[file_md5]
          if old_name == filename:
              return "[跳过]数据已上传（该文件已存在于知识库）"
          if old_name:
              return f"[跳过]数据已上传（内容与已有文件 {old_name} 完全相同）"
          return "[跳过]数据已上传（内容已存在于知识库）"

      # 同名但内容不同 -> 替换语义：先删该文件全部旧向量，再写入新向量
      old_md5 = find_md5_by_filename(records, filename)
      if old_md5 is not None:
          self.chroma.delete(where={"source": filename})
          remove_md5_record(old_md5)

      # ---- 解析 -> 清洗 -> 分割 ----
      try:
          chunks = process(data, file_type)
      except Exception as e:
          return f"[失败]文件解析失败：{e}"

      # ---- chunk 级去重：已存在的块跳过 embedding----
      chunk_hashes = [chunk_md5(c) for c in chunks]
      existing_metas = self.chroma.get(
          where={"hash": {"$in": chunk_hashes}}, include=["metadatas"]
      )["metadatas"]
      existing_hashes = {m["hash"] for m in existing_metas if m}

      new_chunks, new_hashes = [], []
      for text, h in zip(chunks, chunk_hashes):
          if h not in existing_hashes:
              new_chunks.append(text)
              new_hashes.append(h)

      if not new_chunks:
          # 所有文本块都已存在（文件指纹不同但内容实际重复）
          save_md5_record(file_md5, filename)
          return "[跳过]数据已上传（所有文本块均已存在于知识库）"

      # ---- 元数据 + 入库 ----
      now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      metadatas = [
          {
              "source": filename,        # 来源文件
              "create_time": now,        # 上传时间
              "file_type": file_type,    # 原文件格式
              "file_size": len(data),    # 原文件大小（字节）
              "chunk_index": i,          # 该块在文件内的序号
              "hash": h,                 # 块指纹（chunk 级去重依据）
          }
          for i, h in enumerate(new_hashes)
      ]
      self.chroma.add_texts(new_chunks, metadatas=metadatas)
      save_md5_record(file_md5, filename)

      action = "已替换旧版本" if old_md5 is not None else "已上传"
      return (f"[成功]{filename} {action}：共 {len(chunks)} 个文本块，"
              f"新嵌入 {len(new_chunks)} 块，跳过重复 {len(chunks) - len(new_chunks)} 块")

    def uploader_by_str(self, data: str, filename: str) -> str:
      """兼容旧入口：纯文本内容直接上传（按 utf-8 编码后走统一管线）"""
      return self.uploader_file(data.encode("utf-8"), filename)

    def list_files(self) -> list[dict]:
        """列出知识库中所有文件的统计信息，按上传时间倒序"""
        # Chroma 没有"列出全部 source"的API，标准做法：取全部元数据后按文件名分组
        all_metas = self.chroma.get(include=["metadatas"])["metadatas"]

        file_map = {}
        for m in all_metas:
            if not m or "source" not in m:
                continue
            key = m["source"]
            # setdefault：第一次见到该文件时创建统计字典，之后只累加
            info = file_map.setdefault(key, {
                "source": key,
                "chunk_count": 0,
                "create_time": m.get("create_time", ""),
                "file_type": m.get("file_type", ""),
                "file_size": m.get("file_size", 0),
            })
            info["chunk_count"] += 1
            # 取最早的 create_time作该文件的上传时间
            if m.get("create_time", "") and m["create_time"] < info["create_time"]:
                info["create_time"] = m["create_time"]

        files = list(file_map.values())
        # "%Y-%m-%d %H:%M:%S" 格式的字符串，字典序 = 时间序，可直接比较
        files.sort(key=lambda f: f["create_time"], reverse=True)
        return files


    def delete_file(self, filename: str) -> str:
        """删除知识库中某个文件的全部向量，并同步清理文件级 md5 记录"""
        # 与上传入口同样的文件名安全处理
        filename = filename.replace("\\", "/").split("/")[-1]

        # 1. 先查后删：delete对不存在的文件静默无效果，先确认才能给出准确提示
        existing = self.chroma.get(where={"source": filename})
        if not existing["ids"]:
            return f"[失败]知识库中不存在文件 {filename}"

        # 2. 删除该文件的全部向量
        self.chroma.delete(where={"source": filename})

        # 3. 同步清理文件级 md5 记录，防止旧指纹变成"幽灵记录"
        # 不清理的话，之后重新上传同名文件会被误判为"替换旧版本"而非首次上传
        records = read_md5_records()
        old_md5 = find_md5_by_filename(records, filename)
        if old_md5 is not None:
            remove_md5_record(old_md5)

        return f"[成功]文件 {filename} 已删除（共移除{len(existing['ids'])}个文本块）"


if __name__ == "__main__":
  service = KnowledgeBaseService()
  print(service.uploader_by_str("周杰林222", "testfile.txt"))