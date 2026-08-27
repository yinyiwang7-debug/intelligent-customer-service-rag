import re
import hashlib
from io import BytesIO

import config_data as config
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ================= 1. 解析：不同格式 -> 纯文本 =================

def _decode_bytes(raw: bytes) -> str:
    """编码探测：先试 utf-8，失败再 gbk，最后忽略异常字节兜底"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_text(raw: bytes) -> str:
    """txt / markdown 本质都是纯文本，直接解码"""
    return _decode_bytes(raw)


def parse_pdf(raw: bytes) -> str:
    """pdf：pypdf 逐页提取文本"""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def parse_docx(raw: bytes) -> str:
    """docx：提取段落 + 表格"""
    from docx import Document

    doc = Document(BytesIO(raw))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


# 扩展名 -> 解析函数 查表
PARSERS = {
    "txt": parse_text,
    "md": parse_text,
    "pdf": parse_pdf,
    "docx": parse_docx,
}


# ================= 2. 清洗数据 =================

def _is_page_number(line: str) -> bool:
    """整行只由数字、空格、点、短横组成 -> 判定为页码行"""
    return bool(re.fullmatch(r"[\d\s.\-·]+", line)) and any(c.isdigit() for c in line)


def clean_text(text: str) -> str:
    """清洗管线：控制字符 -> 页码行 -> 跨页重复行（页眉页脚） -> 空行压缩"""
    # 控制字符清理（保留 \n 和 \t）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    lines = text.split("\n")

    if config.clean_page_number:
        # 删除纯数字行
        lines = [ln for ln in lines if not _is_page_number(ln)]

    if config.clean_header_footer:
        # 启发式：全文中重复出现 >=3 次且长度 <=30 的行，视为页眉/页脚特征行删除
        # 代价：正常文档里高频短行也会被删
        counter = {}
        for ln in lines:
            key = ln.strip()
            if key:
                counter[key] = counter.get(key, 0) + 1
        repeat_lines = {k for k, v in counter.items() if v >= 3 and len(k) <= 30}
        lines = [ln for ln in lines if ln.strip() not in repeat_lines]

    text = "\n".join(lines)

    if config.compress_blank:
        # 空行压缩
        text = re.sub(r"\n{3,}", "\n\n", text)  # 3 个及以上连续换行 -> 1 个空行

    return text.strip()


# ================= 3. 分割：按文件类型动态选择策略 =================

def split_text(text: str, file_type: str) -> list[str]:
    """动态分割入口：策略按 file_type 查 config.splitter_config，代码里不写死"""
    if len(text) <= config.max_spliter_char_number:
        return [text]  # 短文本不分割，整体作为一个块

    cfg = config.splitter_config[file_type]

    if file_type == "md":
        # markdown 两段式：先按标题切结构，再对超长段按长度切
        # strip_headers=False：标题行保留在块文本里，检索时保留上下文
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=cfg["headers_to_split_on"],
            strip_headers=False,
        )
        length_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=cfg["separators"],
            length_function=len,
        )
        chunks = []
        # split_text 返回 Document 对象列表：文本在 page_content，标题层级在 metadata
        for piece in header_splitter.split_text(text):
            piece_text = piece.page_content
            if len(piece_text) <= config.max_spliter_char_number:
                chunks.append(piece_text)
            else:
                chunks.extend(length_splitter.split_text(piece_text))
        return chunks

    # txt / pdf / docx：统一按长度递归分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=cfg["separators"],
        length_function=len,
    )
    return splitter.split_text(text)


# ================= 4. chunk 哈希（chunk 级去重用） =================

def chunk_md5(text: str) -> str:
    """文本块指纹：先 strip 归一化再哈希，纯空白差异不产生不同指纹"""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


# ================= 5. 管线主入口 =================

def process(raw: bytes, file_type: str) -> list[str]:
    """完整管线：解析 -> 清洗 -> 分割，返回文本块列表"""
    parse_func = PARSERS[file_type]
    text = parse_func(raw)
    if not text.strip():
        raise ValueError("未提取到文本内容（可能是空文件或扫描版 PDF）")
    text = clean_text(text)
    return split_text(text, file_type)