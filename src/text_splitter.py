import json
import time
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Tuple
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

def debug_print(msg): print(f"[DEBUG] {msg}")
def info_print(msg): print(f"[INFO] {msg}")
def error_print(msg): print(f"[ERROR] {msg}")

class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100,
                 min_chunk_tokens: int = 50, encoding_name: str = "o200k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " ", ""]
        )
        debug_print(f"初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, min_chunk_tokens={min_chunk_tokens}")

    def _load_subset_mapping(self, subset_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
        mapping = {}
        if not subset_csv or not subset_csv.exists():
            print(f"[WARNING] 未找到任何 {subset_csv}文件")
            return mapping
        try:
            df = pd.read_csv(subset_csv, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(subset_csv, encoding='gbk')
        for _, row in df.iterrows():
            raw_key = str(row.get("file_name", row.get("sha1", "")))
            clean_key = raw_key.rsplit(".", 1)[0] if "." in raw_key else raw_key
            mapping[clean_key] = {
                "company_name": str(row.get("company_name", "")),
                "sha1": str(row.get("sha1", "")),
                "file_name": raw_key
            }
        return mapping

    # ---------- 辅助函数 ----------
    def _add_page_to_list(self, pages: List[int], page_num: int) -> List[int]:
        if page_num not in pages:
            pages.append(page_num)
            pages.sort()
        return pages

    # ---------- 文本分块（支持 pages 列表）----------
    def _chunk_text(self, text: str, metadata: Dict[str, Any], pages: List[int]) -> Iterator[Dict[str, Any]]:
        if not text or len(text) < 5:
            return
        est_tokens = int(len(text) * 1.4)
        if est_tokens <= self.chunk_size:
            yield {"content": text, "metadata": {**metadata, "pages": pages, "token_len": est_tokens}}
            return
        exact_tokens = len(self.tokenizer.encode(text))
        if exact_tokens <= self.chunk_size:
            yield {"content": text, "metadata": {**metadata, "pages": pages, "token_len": exact_tokens}}
            return
        splits = self.splitter.split_text(text)
        for i, chunk_text in enumerate(splits):
            meta = metadata.copy()
            meta.update({
                "chunk_index": i,
                "chunk_total": len(splits),
                "pages": pages,
                "token_len": len(self.tokenizer.encode(chunk_text))
            })
            yield {"content": chunk_text, "metadata": meta}

    def _extract_text_from_block(self, block: Dict) -> Optional[str]:
        bt = block.get("type", "")
        if bt in ("text", "header", "footer", "page_number", "aside_text", "page_footnote"):
            return block.get("text", "")
        elif bt == "list":
            items = block.get("list_items", [])
            return "\n".join(str(i) for i in items) if isinstance(items, list) else ""
        elif bt == "table":
            return block.get("table_body", "")
        elif bt in ("image", "chart"):
            parts = []
            if block.get("image_caption"):
                parts.append(" ".join(str(c) for c in block["image_caption"]))
            if block.get("image_footnote"):
                parts.append(" ".join(str(f) for f in block["image_footnote"]))
            if block.get("img_path"):
                parts.append(f"[图片: {block['img_path']}]")
            return " ".join(parts) if parts else ""
        elif bt == "equation":
            return block.get("text", "") or block.get("img_path", "")
        return block.get("text", "")

    def _update_title_stack(self, title_text: str, level: int, stack: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        while stack and stack[-1][1] >= level:
            stack.pop()
        stack.append((title_text, level))
        return stack

    def _get_full_title_path(self, stack: List[Tuple[str, int]]) -> str:
        return " > ".join([t[0] for t in stack]) if stack else ""

    @staticmethod
    def _html_table_to_markdown(html: str) -> str:
        """
        将 HTML 表格转换为 Markdown 表格，支持 rowspan 和 colspan。
        对于合并的单元格，通过重复内容来填充。
        """
        import re
        from html.parser import HTMLParser

        if not html or "<table" not in html.lower():
            return html

        # ---------- 1. 解析表格 ----------
        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_table = False
                self.in_row = False
                self.in_cell = False
                self.current_row = []
                self.rows = []
                self.current_cell_text = []
                self.current_rowspan = 1
                self.current_colspan = 1

            def handle_starttag(self, tag, attrs):
                if tag == 'table':
                    self.in_table = True
                    self.rows = []
                elif self.in_table and tag == 'tr':
                    self.in_row = True
                    self.current_row = []
                elif self.in_table and self.in_row and tag in ('td', 'th'):
                    self.in_cell = True
                    self.current_cell_text = []
                    attrs_dict = dict(attrs)
                    self.current_rowspan = int(attrs_dict.get('rowspan', 1))
                    self.current_colspan = int(attrs_dict.get('colspan', 1))

            def handle_endtag(self, tag):
                if tag == 'table':
                    self.in_table = False
                elif tag == 'tr':
                    if self.in_row:
                        self.rows.append(self.current_row)
                        self.in_row = False
                elif tag in ('td', 'th') and self.in_cell:
                    cell_text = ''.join(self.current_cell_text).strip()
                    self.current_row.append({
                        'text': cell_text,
                        'rowspan': self.current_rowspan,
                        'colspan': self.current_colspan
                    })
                    self.in_cell = False

            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell_text.append(data)

        parser = TableParser()
        try:
            parser.feed(html)
        except Exception:
            return TextSplitter._simple_html_table_to_markdown(html)

        rows = parser.rows
        if not rows:
            return html

        # ---------- 2. 计算最大列数 ----------
        max_cols = 0
        for row in rows:
            col_idx = 0
            for cell in row:
                col_idx += cell['colspan']
            max_cols = max(max_cols, col_idx)

        # ---------- 3. 构建网格，标记合并单元格 ----------
        # 初始化网格
        grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

        # 存储每个单元格的原始信息
        cell_info = [[None for _ in range(max_cols)] for _ in range(len(rows))]

        for i, row in enumerate(rows):
            col_idx = 0
            for cell in row:
                # 跳过已被占用的列
                while col_idx < max_cols and grid[i][col_idx] is not None:
                    col_idx += 1
                if col_idx >= max_cols:
                    break

                rowspan = cell['rowspan']
                colspan = cell['colspan']
                text = cell['text']

                # 存储单元格
                grid[i][col_idx] = text
                cell_info[i][col_idx] = {
                    'text': text,
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'is_start': True
                }

                # 🔑 标记被 colspan 占用的位置（同行后续列）
                for c in range(1, colspan):
                    if col_idx + c < max_cols:
                        grid[i][col_idx + c] = ''  # 临时占位
                        cell_info[i][col_idx + c] = {
                            'is_placeholder': True,
                            'source_row': i,
                            'source_col': col_idx
                        }

                # 🔑 标记被 rowspan 占用的位置（后续行）
                for r in range(1, rowspan):
                    if i + r < len(grid):
                        for c in range(colspan):
                            if col_idx + c < max_cols:
                                grid[i + r][col_idx + c] = ''  # 临时占位
                                cell_info[i + r][col_idx + c] = {
                                    'is_placeholder': True,
                                    'source_row': i,
                                    'source_col': col_idx
                                }

                col_idx += colspan

        # ---------- 4. 🔑 关键：填充所有占位符（重复内容）----------
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if cell_info[i][j] and cell_info[i][j].get('is_placeholder'):
                    src_row = cell_info[i][j]['source_row']
                    src_col = cell_info[i][j]['source_col']
                    # 从源单元格复制内容
                    if src_row < len(grid) and src_col < len(grid[src_row]):
                        grid[i][j] = grid[src_row][src_col]

        # ---------- 5. 将 None 转换为空字符串 ----------
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] is None:
                    grid[i][j] = ''

        # ---------- 6. 🔑 修复版 clean_cell：正确处理换行符 ----------
        def clean_cell(txt):
            if not txt:
                return ''
            # 将换行符替换为空格（保持内容连贯）
            txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # 合并多个空格
            txt = re.sub(r'\s+', ' ', txt).strip()
            # 转义管道符
            txt = txt.replace('|', '\\|')
            return txt

        # ---------- 7. 构建 Markdown 表格 ----------
        md_rows = []

        # 处理表头（第一行作为表头）
        header_cells = [clean_cell(cell) for cell in grid[0]]
        md_rows.append('| ' + ' | '.join(header_cells) + ' |')

        # 分隔行
        separator = '| ' + ' | '.join(['---'] * len(grid[0])) + ' |'
        md_rows.append(separator)

        # 数据行（从第1行开始）
        for row in grid[1:]:
            md_cells = [clean_cell(cell) for cell in row]
            # 跳过全空行
            if all(not c or c == '' for c in md_cells):
                continue
            md_rows.append('| ' + ' | '.join(md_cells) + ' |')

        return '\n'.join(md_rows)

    @staticmethod
    def _simple_html_table_to_markdown(html: str) -> str:
        """降级方案：简单提取表格行和单元格，忽略 rowspan/colspan"""
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        if not rows:
            return html

        all_cells = []
        for row_html in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
            if not cells:
                continue
            clean_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if all(not c for c in clean_cells):
                continue
            all_cells.append(clean_cells)

        if not all_cells:
            return html

        max_cols = max(len(r) for r in all_cells)
        for r in all_cells:
            r.extend([""] * (max_cols - len(r)))

        md_lines = [f"| {' | '.join(row)} |" for row in all_cells]
        if len(md_lines) > 1:
            md_lines.insert(1, f"| {' | '.join(['---'] * max_cols)} |")
        return "\n".join(md_lines)

    def _parse_table_content(self, content: str) -> Dict[str, Any]:
        default = {"row_count": 0, "col_count": 0, "cells": [], "parse_status": "unparseable"}
        if not content or not isinstance(content, str):
            return default
        cells = []
        status = "unparseable"
        try:
            if "<table" in content or "<tr" in content:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL | re.IGNORECASE)
                if rows:
                    for r_idx, row in enumerate(rows):
                        cell_matches = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
                        for c_idx, raw in enumerate(cell_matches):
                            txt = re.sub(r'<[^>]+>', '', raw).strip()
                            if txt:
                                cells.append({"row": r_idx, "col": c_idx, "text": txt})
                    status = "success" if cells else "fallback_html"
            if not cells and "|" in content:
                lines = [l.strip() for l in content.splitlines() if l.strip() and not set(l.replace(" ", "")).issubset(set("-|"))]
                for r_idx, line in enumerate(lines):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    for c_idx, txt in enumerate(parts):
                        cells.append({"row": r_idx, "col": c_idx, "text": txt})
                    status = "success" if cells else "fallback_md"
            if not cells:
                for r_idx, line in enumerate([l.strip() for l in content.splitlines() if l.strip()]):
                    cells.append({"row": r_idx, "col": 0, "text": line})
                status = "fallback_plain"
        except Exception as e:
            debug_print(f"表格解析异常: {e}")
            status = "error"
        return {
            "row_count": max((c["row"]+1 for c in cells), default=0),
            "col_count": max((c["col"]+1 for c in cells), default=0),
            "cells": cells,
            "parse_status": status
    }

    # ---------- 后处理合并短块（保留 pages）----------
    def _merge_short_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks
        merged = []
        for ch in chunks:
            if not merged:
                merged.append(ch)
                continue
            last = merged[-1]
            cur_tok = ch['metadata'].get('token_len', 0)
            if (cur_tok < self.min_chunk_tokens and
                ch['metadata'].get('block_type') == last['metadata'].get('block_type') and
                ch['metadata'].get('section_title') == last['metadata'].get('section_title') and
                ch['metadata'].get('text_level', 0) == 0 and
                last['metadata'].get('text_level', 0) == 0 and
                ch['metadata'].get('block_type') not in ('表格', '图片', '图表', '公式') and
                last['metadata'].get('block_type') not in ('表格', '图片', '图表', '公式')):
                last['content'] += " " + ch['content']
                last['metadata']['token_len'] += cur_tok
                # 合并页码
                last_pages = last['metadata'].get('pages', [])
                cur_pages = ch['metadata'].get('pages', [])
                last['metadata']['pages'] = sorted(set(last_pages + cur_pages))
            else:
                merged.append(ch)
        return merged

    # ---------- 核心：带缓冲区的处理（支持跨页记录，页眉/页脚不打断）----------
    def _process_single_json(self, json_path: Path, meta_base: Dict) -> Iterator[Dict]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            error_print(f"{json_path.name} 顶层不是列表，跳过")
            return

        type_map = {
            "text": "正文", "title": "标题", "header": "页眉", "page_number": "页码",
            "list": "列表", "table": "表格", "image": "图片", "chart": "图表",
            "equation": "公式", "footer": "页脚", "aside_text": "侧边文本", "page_footnote": "页面脚注"
        }

        title_stack = []
        current_section_path = ""
        current_section_level = 0

        # 缓冲区
        buffer_texts = []
        buffer_pages = []
        buffer_metadata = None
        buffer_type = None

        def flush_buffer():
            nonlocal buffer_texts, buffer_pages, buffer_metadata, buffer_type
            if buffer_texts:
                combined = "\n\n".join(buffer_texts)
                pages = sorted(set(buffer_pages))
                yield from self._chunk_text(combined, buffer_metadata, pages)
                buffer_texts = []
                buffer_pages = []
                buffer_metadata = None
                buffer_type = None

        for block in data:
            bt = block.get("type", "unknown")
            tl = block.get("text_level", 0)

            # 跳过页眉、页脚、页码（不加入缓冲区，不刷新）
            if bt in ("header", "footer", "page_number"):
                continue

            text = self._extract_text_from_block(block)
            if not text:
                continue

            page_num = block.get("page_idx", 0) + 1
            bbox = block.get("bbox", [])
            block_id = hashlib.md5(f"{json_path.name}_{page_num}_{bbox}".encode()).hexdigest()
            base_meta = {
                "source_file": json_path.name,
                "coordinates": bbox,
                "block_id": block_id,
                **meta_base
            }

            # 标题：更新栈，加入缓冲区
            if bt == "text" and tl >= 1:
                title_stack = self._update_title_stack(text, tl, title_stack)
                current_section_path = self._get_full_title_path(title_stack)
                current_section_level = tl
                if buffer_metadata is None:
                    buffer_metadata = {
                        **base_meta,
                        "block_type": "正文",
                        "text_level": 0,
                        "section_title": current_section_path,
                        "section_level": current_section_level
                    }
                    buffer_type = "text"
                buffer_texts.append(text)
                buffer_pages = self._add_page_to_list(buffer_pages, page_num)
                continue

            # 判断是否可合并（正文/列表）
            can_merge = (bt in ("text", "list") and
                         (buffer_type is None or buffer_type == "text"))

            if can_merge:
                if buffer_metadata is None:
                    buffer_metadata = {
                        **base_meta,
                        "block_type": "正文",
                        "text_level": 0,
                        "section_title": current_section_path,
                        "section_level": current_section_level
                    }
                    buffer_type = "text"
                buffer_texts.append(text)
                buffer_pages = self._add_page_to_list(buffer_pages, page_num)
            else:
                # 不能合并，先刷新缓冲区
                yield from flush_buffer()
                # 单独处理当前块
                if bt == "table":
                    struct = self._parse_table_content(text)
                    md = self._html_table_to_markdown(text) if "<table" in text else text
                    meta = {
                        **base_meta,
                        "block_type": type_map.get(bt, bt),
                        "text_level": 0,
                        "section_title": current_section_path,
                        "section_level": current_section_level,
                        "token_len": len(self.tokenizer.encode(md)),
                        "table_structure": struct
                    }
                    yield {"content": md.strip(), "metadata": {**meta, "pages": [page_num]}}
                elif bt in ("image", "chart", "equation"):
                    meta = {
                        **base_meta,
                        "block_type": type_map.get(bt, bt),
                        "text_level": 0,
                        "section_title": current_section_path,
                        "section_level": current_section_level,
                        "token_len": len(self.tokenizer.encode(text))
                    }
                    yield {"content": text.strip(), "metadata": {**meta, "pages": [page_num]}}
                else:
                    # 其他类型（如单独的 text 未被合并），也单独输出
                    meta = {
                        **base_meta,
                        "block_type": type_map.get(bt, bt),
                        "text_level": 0,
                        "section_title": current_section_path,
                        "section_level": current_section_level
                    }
                    yield from self._chunk_text(text.strip(), meta, [page_num])

        # 循环结束，刷新缓冲区
        yield from flush_buffer()

    # ---------- 保存 JSON ----------
    def _save_chunks_to_json(self, output_dir: Path, base_name: str, chunks: List[Dict], meta_base: Dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{base_name}_chunks.json"
        metainfo = {
            "sha1": meta_base.get("sha1", ""),
            "company_name": meta_base.get("company_name", ""),
            "file_name": meta_base.get("file_name", f"{base_name}.pdf")
        }
        chunks_list = []
        for ch in chunks:
            chunks_list.append({
                "pages": ch["metadata"].get("pages", []),
                "length_tokens": ch["metadata"].get("token_len", 0),
                "text": ch["content"],
                "type": ch["metadata"].get("block_type", "unknown"),
                "section_title": ch["metadata"].get("section_title", ""),
                "section_level": ch["metadata"].get("section_level", 0),
                "coordinates": ch["metadata"].get("coordinates", []),
                "table_structure": ch["metadata"].get("table_structure")
            })
        output_data = {
            "metainfo": metainfo,
            "content": {"chunks": chunks_list}
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        debug_print(f"保存文件: {out_path}")

    # ---------- 主入口 ----------
    def split_mineru_jsons(self, input_dir: Path, output_dir: Path, subset_csv: Optional[Path] = None) -> None:
        if not input_dir.exists():
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        file2meta = self._load_subset_mapping(subset_csv)
        json_paths = sorted(input_dir.glob("*_content_list.json"))
        if not json_paths:
            print("[WARNING] 未找到任何 *_content_list.json 文件")
            return
        info_print(f"找到 {len(json_paths)} 个 content_list.json 文件")
        total_chunks = 0
        t_start = time.time()
        for idx, jp in enumerate(json_paths, 1):
            file_t0 = time.time()
            raw_chunks = []
            try:
                bn = jp.stem.replace("_content_list", "")
                mb = file2meta.get(bn, {"file_name": f"{bn}.pdf"})
                for c in self._process_single_json(jp, mb):
                    raw_chunks.append(c)
                merged = self._merge_short_chunks(raw_chunks)
                total_chunks += len(merged)
                self._save_chunks_to_json(output_dir, bn, merged, mb)
                elapsed = time.time() - file_t0
                info_print(f"[{idx}/{len(json_paths)}] {jp.name} | {len(raw_chunks)} -> {len(merged)} chunks | {elapsed:.1f}s")
            except Exception as e:
                error_print(f"处理失败 {jp.name}: {e}")
                continue
        info_print(f"完成！总 chunk 数: {total_chunks}，耗时: {time.time()-t_start:.1f}s")
        info_print(f"输出目录: {output_dir.resolve()}")

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))