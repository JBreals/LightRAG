"""
RAG-Anything integration utilities for LightRAG API.

This module provides utilities for integrating RAG-Anything multimodal
document processing capabilities into the LightRAG server.
"""

import asyncio
import base64
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Callable, Any

from lightrag.utils import logger


@lru_cache(maxsize=1)
def _is_raganything_available() -> bool:
    """Check if raganything is available (cached check).

    Returns:
        bool: True if raganything is available, False otherwise
    """
    try:
        import raganything  # noqa: F401

        return True
    except ImportError:
        return False


def get_supported_multimodal_extensions() -> tuple:
    """Get file extensions supported by RAG-Anything multimodal processing.

    Returns:
        tuple: Supported file extensions
    """
    return (
        # Documents
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".ppt",
        ".xls",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    )


def is_image_file(file_path: Path) -> bool:
    """Check if a file is an image.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if the file is an image
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
    return file_path.suffix.lower() in image_extensions


def create_vision_model_func(
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Callable:
    """Create a vision model function for RAG-Anything.

    Args:
        model: Vision model name (e.g., 'gpt-4o', 'gpt-4-vision-preview')
        api_key: API key for the vision model
        api_base: API base URL (optional, for custom endpoints)

    Returns:
        Callable: Async function that processes images with the vision model
    """

    async def vision_model_func(
        prompt: str,
        image_data: str | list[str],
        **kwargs,
    ) -> str:
        """Process image(s) with vision model.

        Args:
            prompt: Text prompt for the vision model
            image_data: Base64 encoded image(s) or image URL(s)
            **kwargs: Additional arguments (system_prompt, max_tokens)

        Returns:
            str: Vision model response
        """
        try:
            from openai import AsyncOpenAI

            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if api_base:
                client_kwargs["base_url"] = api_base

            client = AsyncOpenAI(**client_kwargs)

            # Build message content
            content = [{"type": "text", "text": prompt}]

            # Handle single or multiple images
            images = image_data if isinstance(image_data, list) else [image_data]
            for img in images:
                if img.startswith("http"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                else:
                    # Assume base64 encoded
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}
                    })

            # Build messages with optional system prompt
            messages = []
            system_prompt = kwargs.get("system_prompt")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1024),
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision model error: {e}")
            raise

    return vision_model_func


def _get_vision_system_prompt(language: str) -> str:
    """Get vision model system prompt in the specified language.

    Args:
        language: Target language for the prompt

    Returns:
        str: Vision model system prompt in the specified language
    """
    prompts = {
        "Korean": """당신은 문서 분석 전문가이며, RAG(검색 증강 생성) 시스템에 투입될
이미지 설명 데이터를 생성하는 역할을 맡고 있습니다.

목표:
- 문서에서 추출된 이미지를 분석하여
- 검색 및 추론에 유용한 텍스트 설명을 생성합니다.

작업 절차는 반드시 다음 2단계를 따르세요.

────────────────────
[1단계: 정보 밀도 분류]
────────────────────
먼저 이미지를 분석하여 정보 밀도를 아래 중 하나로 분류하세요.

- SIMPLE:
  - 텍스트가 거의 없거나 매우 짧음
  - 장식용 이미지, 아이콘, 로고
  - 단순한 개념도 또는 반복적인 구조
  - RAG 검색 시 핵심 키워드만 있으면 충분한 경우

- COMPLEX:
  - 다수의 텍스트 요소가 존재
  - 표, 다이어그램, 차트, 프로세스 흐름 포함
  - 문서의 핵심 내용을 전달하는 이미지
  - RAG에서 문맥 이해 및 추론에 중요한 경우

분류 결과는 내부 판단으로만 사용하고,
출력에는 분류 결과를 직접 표시하지 마세요.

────────────────────
[2단계: 분류에 따른 설명 생성]
────────────────────

[SIMPLE로 판단된 경우]
- 3~5문장 이내로 간결하게 설명하세요
- 이미지의 목적과 핵심 의미만 요약하세요
- 불필요한 추론이나 세부 묘사는 하지 마세요
- 명확히 보이지 않는 정보는 추측하지 마세요

[COMPLEX로 판단된 경우]
- 아래 항목을 모두 충족하도록 상세히 설명하세요:
  - 이미지에 포함된 모든 텍스트를 정확히 추출
  - 다이어그램, 표, 차트의 구조와 관계 설명
  - 데이터 흐름, 단계, 계층 구조 명시
  - 문서 전체 맥락에서 이 이미지의 역할 설명
- 기술 용어와 전문 용어는 원문 그대로 유지하세요
- 목록, 단계, 구조는 명확히 구분하여 서술하세요

────────────────────
[공통 규칙]
────────────────────
- 이미지에 없는 정보는 절대 추론하거나 생성하지 마세요
- 불명확한 요소는 “식별 불가” 또는 “텍스트 없음”으로 명시하세요
- 응답은 반드시 한국어로 작성하세요
- RAG 시스템에서 검색 가능한 명확한 표현을 우선하세요
""",

        "Chinese": """您是一位文档分析专家。您分析从文档中提取的图像，并提供详细准确的描述，以便在RAG（检索增强生成）系统中使用。

请遵循以下准则：
- 准确提取图像中包含的所有文本
- 描述图表、图形的结构和数据
- 理解图像的上下文及其在文档中的作用
- 保留技术术语和专业术语
- 用中文回答""",

        "Japanese": """あなたは文書分析の専門家です。文書から抽出された画像を分析し、RAG（検索拡張生成）システムで活用できるよう、詳細で正確な説明を提供します。

以下のガイドラインに従ってください：
- 画像に含まれるすべてのテキストを正確に抽出する
- ダイアグラム、チャート、グラフの構造とデータを説明する
- 画像のコンテキストと文書内での役割を把握する
- 技術用語や専門用語はそのまま維持する
- 日本語で回答する""",
    }
    return prompts.get(
        language,
        """You are a document analysis expert. You analyze images extracted from documents and provide detailed, accurate descriptions for use in RAG (Retrieval-Augmented Generation) systems.

Follow these guidelines:
- Extract all text contained in the image accurately
- Describe the structure and data of diagrams, charts, and graphs
- Understand the context of the image and its role in the document
- Preserve technical terms and terminology
- Respond in English"""
    )


def _get_vision_prompt(language: str) -> str:
    """Get vision model user prompt in the specified language.

    Args:
        language: Target language for the prompt

    Returns:
        str: Vision model user prompt in the specified language
    """
    prompts = {
        "Korean": """이 이미지를 문서의 일부로 간주하고 분석해주세요.

- 먼저 이 이미지의 정보 밀도를 판단한 뒤,
- 그 판단에 따라 적절한 수준의 설명을 제공하세요.

이미지에 포함된 텍스트, 구조, 다이어그램, 표, 차트 등
문서 이해와 검색에 의미 있는 시각적 정보에만 집중하세요.
불필요한 추론이나 과도한 설명은 피하세요.

        """,
        "Chinese": "请分析并描述这张图片，重点关注其中的文字、图表、图形、表格或重要的视觉信息。",
        "Japanese": "この画像を分析して説明してください。テキスト、ダイアグラム、チャート、表、または重要な視覚情報に焦点を当ててください。",
        "German": "Analysieren und beschreiben Sie dieses Bild. Konzentrieren Sie sich auf Text, Diagramme, Grafiken, Tabellen oder wichtige visuelle Informationen.",
        "French": "Analysez et décrivez cette image. Concentrez-vous sur le texte, les diagrammes, les graphiques, les tableaux ou les informations visuelles importantes.",
        "Spanish": "Analiza y describe esta imagen. Céntrate en el texto, diagramas, gráficos, tablas o información visual importante.",
    }
    return prompts.get(
        language,
        "Analyze and describe this image. Focus on any text, diagrams, charts, tables, or important visual information."
    )


def _get_docling_vision_system_prompt(language: str) -> str:
    """Get Docling-optimized vision model system prompt.

    Docling extracts figures from structured documents (papers, reports, manuals).
    These prompts are optimized for technical/scientific document figures.

    Args:
        language: Target language for the prompt

    Returns:
        str: Docling-optimized system prompt
    """
    prompts = {
        "Korean": """당신은 기술 문서 분석 전문가입니다. 학술 논문, 기술 보고서, 매뉴얼에서 추출된 그림(figure)을 분석합니다.

분석 시 다음 사항을 포함하세요:
1. **그림 유형 식별**: 다이어그램, 플로우차트, 아키텍처도, 그래프, 스크린샷 등
2. **핵심 구성요소**: 주요 요소, 레이블, 범례, 축 정보
3. **데이터 및 수치**: 그래프의 경우 트렌드, 수치, 단위 포함
4. **관계 및 흐름**: 화살표, 연결선이 나타내는 프로세스나 관계
5. **문서 맥락**: 이 그림이 문서에서 설명하려는 개념

기술 용어는 원어 그대로 유지하고, 한국어로 상세히 설명하세요.""",

        "Chinese": """您是技术文档分析专家。您分析从学术论文、技术报告和手册中提取的图表。

分析时请包含以下内容：
1. **图表类型识别**：图表、流程图、架构图、图形、截图等
2. **核心组件**：主要元素、标签、图例、轴信息
3. **数据和数值**：对于图表，包括趋势、数值、单位
4. **关系和流程**：箭头、连接线表示的过程或关系
5. **文档上下文**：这张图在文档中要解释的概念

保留技术术语原文，用中文详细说明。""",

        "Japanese": """あなたは技術文書分析の専門家です。学術論文、技術レポート、マニュアルから抽出された図を分析します。

分析には以下を含めてください：
1. **図の種類の特定**：ダイアグラム、フローチャート、アーキテクチャ図、グラフ、スクリーンショットなど
2. **主要コンポーネント**：主要要素、ラベル、凡例、軸情報
3. **データと数値**：グラフの場合、トレンド、数値、単位を含める
4. **関係とフロー**：矢印、接続線が示すプロセスや関係
5. **文書コンテキスト**：この図が文書で説明しようとしている概念

技術用語は原語のまま維持し、日本語で詳しく説明してください。""",
    }
    return prompts.get(
        language,
        """You are a technical document analysis expert. You analyze figures extracted from academic papers, technical reports, and manuals.

Include the following in your analysis:
1. **Figure Type**: Identify if it's a diagram, flowchart, architecture diagram, graph, chart, screenshot, etc.
2. **Key Components**: Main elements, labels, legends, axis information
3. **Data & Values**: For graphs/charts, include trends, specific values, units
4. **Relationships & Flow**: Describe what arrows and connections represent
5. **Document Context**: What concept this figure is illustrating

Preserve technical terms as-is and provide a detailed, structured description."""
    )


def _get_docling_vision_prompt(language: str) -> str:
    """Get Docling-optimized vision model user prompt.

    Args:
        language: Target language for the prompt

    Returns:
        str: Docling-optimized user prompt
    """
    prompts = {
        "Korean": """이 문서 그림을 분석해주세요.

다음 형식으로 응답하세요:
- **유형**: [그림 유형]
- **구성요소**: [주요 요소 나열]
- **설명**: [상세 설명]
- **핵심 정보**: [문서에서 이 그림이 전달하는 핵심 메시지]""",

        "Chinese": """请分析这张文档图表。

请按以下格式回答：
- **类型**：[图表类型]
- **组件**：[主要元素列表]
- **描述**：[详细说明]
- **关键信息**：[这张图在文档中传达的核心信息]""",

        "Japanese": """この文書の図を分析してください。

以下の形式で回答してください：
- **タイプ**: [図の種類]
- **コンポーネント**: [主要要素のリスト]
- **説明**: [詳細な説明]
- **重要情報**: [この図が文書で伝えている核心メッセージ]""",
    }
    return prompts.get(
        language,
        """Analyze this document figure.

Respond in the following format:
- **Type**: [Figure type - diagram/flowchart/graph/chart/screenshot/etc.]
- **Components**: [List main elements, labels, legends]
- **Description**: [Detailed description of what the figure shows]
- **Key Information**: [The core message this figure conveys in the document]"""
    )


def _convert_docling_table_to_markdown(table_body: dict) -> str:
    """Convert Docling table structure to markdown format.

    Docling returns table_body as a dict with:
    - table_cells: list of cell data
    - num_rows: number of rows
    - num_cols: number of columns
    - grid: 2D grid representation (each cell is a dict with 'text' key)

    Args:
        table_body: Docling table body dict

    Returns:
        str: Markdown formatted table
    """
    try:
        grid = table_body.get("grid", [])
        num_rows = table_body.get("num_rows", 0)
        num_cols = table_body.get("num_cols", 0)

        if not grid or num_rows == 0 or num_cols == 0:
            # Fallback to table_cells if grid is empty
            cells = table_body.get("table_cells", [])
            if not cells:
                return ""

            # Build grid from cells
            text_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            for cell in cells:
                row = cell.get("row", 0)
                col = cell.get("col", 0)
                text = cell.get("text", "")
                if 0 <= row < num_rows and 0 <= col < num_cols:
                    text_grid[row][col] = text
            grid = text_grid

        # Convert grid to markdown
        md_lines = []
        for i, row in enumerate(grid):
            # Extract text from cell (Docling cells are dicts with 'text' key)
            cell_texts = []
            for cell in row:
                if isinstance(cell, dict):
                    text = cell.get("text", "")
                else:
                    text = str(cell) if cell else ""
                # Escape pipe characters and clean up
                text = text.replace("|", "\\|").replace("\n", " ").strip()
                cell_texts.append(text)

            md_lines.append("| " + " | ".join(cell_texts) + " |")

            # Add header separator after first row
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")

        return "\n".join(md_lines)

    except Exception as e:
        # If conversion fails, return string representation
        return f"[Table conversion error: {e}]"


async def extract_content_with_raganything(
    file_path: Path,
    working_dir: str,
    vision_model_func: Optional[Callable] = None,
    enable_image: bool = True,
    enable_table: bool = True,
    enable_equation: bool = True,
    parser: str = "mineru",
    parse_method: str = "auto",
    language: str = "English",
    ocr_lang: str = "",
    custom_system_prompt: str = "",
    custom_user_prompt: str = "",
    block_mapping_storage: Optional[Any] = None,
) -> str:
    """Extract text content from a file using RAG-Anything (preprocessing only).

    This function uses RAG-Anything to parse multimodal documents and extract
    text content WITHOUT inserting to LightRAG. The extracted text can then
    be passed to the existing LightRAG pipeline.

    Args:
        file_path: Path to the file to process
        working_dir: Working directory for RAG-Anything cache
        vision_model_func: Function for processing images with vision model
        enable_image: Enable image processing
        enable_table: Enable table processing
        enable_equation: Enable equation processing
        parser: Document parser ('mineru' or 'docling')
        parse_method: Parsing method ('auto', 'ocr', 'txt')
        language: Language for image descriptions (default: English)
        ocr_lang: OCR language for MinerU (e.g., 'korean', 'en', 'ch', 'japan')
        custom_system_prompt: Custom system prompt for VLM (empty to use default)
        custom_user_prompt: Custom user prompt for VLM (empty to use default)
        block_mapping_storage: Optional storage for saving block mappings

    Returns:
        str: Extracted text content from the multimodal document

    Raises:
        ImportError: If raganything is not installed
        Exception: If processing fails
    """
    if not _is_raganything_available():
        raise ImportError(
            "RAG-Anything is not installed. "
            "Install it with: pip install raganything"
        )

    from raganything import RAGAnything, RAGAnythingConfig
    from datetime import datetime

    # Import block mapping classes if storage is provided
    if block_mapping_storage is not None:
        from lightrag.kg.json_block_mapping_impl import (
            BlockMapping,
            DocumentBlockMappings,
        )

    try:
        # Create RAG-Anything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=parser,
            parse_method=parse_method,
            enable_image_processing=enable_image,
            enable_table_processing=enable_table,
            enable_equation_processing=enable_equation,
        )

        # Initialize RAG-Anything (without LightRAG instance for parsing only)
        rag_anything = RAGAnything(
            config=config,
            vision_model_func=vision_model_func,
        )

        # Log the actual parse mode that will be used (for MinerU auto detection)
        actual_mode = parse_method
        if parser == "mineru" and parse_method == "auto":
            try:
                from magic_pdf.data.dataset import PymuDocDataset
                from magic_pdf.config.enums import SupportedPdfParseMethod

                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                ds = PymuDocDataset(pdf_bytes)
                classified = ds.classify()
                if classified == SupportedPdfParseMethod.OCR:
                    actual_mode = "auto → OCR (scanned/image PDF detected)"
                else:
                    actual_mode = "auto → TXT (text-based PDF detected)"
            except Exception as e:
                actual_mode = f"auto (classification check failed: {e})"

        logger.info(f"[RAG-Anything] Starting {parser.upper()} parsing for {file_path.name} [mode: {actual_mode}]")
        parse_start_time = time.time()

        # Parse document to get content blocks (without inserting to LightRAG)
        parse_kwargs = {
            "file_path": str(file_path),
            "parse_method": parse_method,
            "display_stats": False,
        }
        # Add OCR language if specified (for MinerU OCR optimization)
        if ocr_lang:
            parse_kwargs["lang"] = ocr_lang
            logger.info(f"[RAG-Anything] OCR language set to: {ocr_lang}")

        content_list, doc_id = await rag_anything.parse_document(**parse_kwargs)

        parse_elapsed = time.time() - parse_start_time

        # Count content types for logging
        type_counts = {}
        for item in content_list:
            t = item.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        logger.info(
            f"[RAG-Anything] Parsing completed in {parse_elapsed:.1f}s - "
            f"{len(content_list)} blocks: {type_counts}"
        )

        # Initialize block mappings collection if storage is provided
        block_mappings: list = [] if block_mapping_storage else None
        stats = {
            "text_blocks": 0,
            "image_blocks": 0,
            "table_blocks": 0,
            "equation_blocks": 0,
            "failed_blocks": 0,
        }

        # Convert content blocks to text
        text_parts = []
        processed_count = 0
        image_count = 0

        for block_idx, item in enumerate(content_list):
            content_type = item.get("type", "")
            page_idx = item.get("page_idx", 0)
            bbox = item.get("bbox", [0, 0, 0, 0])

            # Initialize block mapping variables
            original_content = ""
            converted_content = ""
            processing_time = None
            status = "success"
            error_message = None

            if content_type == "text":
                text = item.get("text", "")
                original_content = text[:500] + "..." if len(text) > 500 else text
                if text.strip():
                    text_parts.append(text)
                    # Store actual text content (truncated for very long texts)
                    converted_content = text[:2000] + "..." if len(text) > 2000 else text
                    stats["text_blocks"] += 1
                else:
                    status = "skipped"
                    converted_content = "[empty]"

            elif content_type == "table" and enable_table:
                # Include table as markdown
                table_body = item.get("table_body", "")
                table_caption = item.get("table_caption", [])

                # Handle docling dict format for table_body
                if isinstance(table_body, dict):
                    # Docling returns table_body as dict with grid/cells
                    table_md = _convert_docling_table_to_markdown(table_body)
                    table_body_str = table_md
                else:
                    table_body_str = table_body

                # Handle caption (can be list or string)
                if isinstance(table_caption, list):
                    caption_text = " ".join(table_caption) if table_caption else ""
                else:
                    caption_text = table_caption or ""

                original_content = f"caption: {caption_text}, body: {table_body_str[:300] if table_body_str else ''}..."

                if table_body_str:
                    if caption_text:
                        output = f"**Table: {caption_text}**\n{table_body_str}"
                    else:
                        output = f"**Table:**\n{table_body_str}"
                    text_parts.append(output)
                    # Store actual markdown table output (truncated for very large tables)
                    converted_content = output[:3000] + "..." if len(output) > 3000 else output
                    stats["table_blocks"] += 1
                else:
                    status = "skipped"
                    converted_content = "[empty table]"

            elif content_type == "equation" and enable_equation:
                # Include equation description
                eq_text = item.get("text", "")
                original_content = eq_text[:500] if eq_text else ""
                if eq_text:
                    output = f"[Equation: {eq_text}]"
                    text_parts.append(output)
                    # Store actual equation output
                    converted_content = output
                    stats["equation_blocks"] += 1
                else:
                    status = "skipped"
                    converted_content = "[empty equation]"

            elif content_type == "image" and enable_image and vision_model_func:
                # Process image with vision model if available
                img_path = item.get("img_path", "")
                original_content = img_path
                if img_path and Path(img_path).exists():
                    image_count += 1
                    total_images = type_counts.get("image", 0)
                    logger.info(
                        f"[RAG-Anything] Processing image {image_count}/{total_images} with vision model..."
                    )
                    img_start_time = time.time()
                    try:
                        img_b64 = encode_image_to_base64(Path(img_path))
                        # Use custom prompts if provided, otherwise use language-based defaults
                        vision_prompt = custom_user_prompt if custom_user_prompt else _get_vision_prompt(language)
                        system_prompt = custom_system_prompt if custom_system_prompt else _get_vision_system_prompt(language)
                        description = await vision_model_func(
                            vision_prompt,
                            img_b64,
                            system_prompt=system_prompt,
                        )
                        # Include caption if available
                        img_caption = item.get("image_caption", [])
                        if img_caption:
                            caption_text = " ".join(img_caption)
                            output = f"[Image: {caption_text}]\n{description}"
                        else:
                            output = f"[Image]\n{description}"
                        text_parts.append(output)
                        img_elapsed = time.time() - img_start_time
                        processing_time = img_elapsed
                        # Store actual VLM output (full description)
                        converted_content = output
                        stats["image_blocks"] += 1
                        logger.info(f"[RAG-Anything] Image {image_count} processed in {img_elapsed:.1f}s")
                    except Exception as e:
                        img_elapsed = time.time() - img_start_time
                        processing_time = img_elapsed
                        status = "failed"
                        error_message = str(e)
                        stats["failed_blocks"] += 1
                        logger.warning(f"[RAG-Anything] Image {image_count} failed after {img_elapsed:.1f}s: {e}")
                        # Still include caption if available
                        img_caption = item.get("image_caption", [])
                        if img_caption:
                            caption_output = f"[Image: {' '.join(img_caption)}]"
                            text_parts.append(caption_output)
                            converted_content = f"[VLM failed, caption only]\n{caption_output}"
                        else:
                            converted_content = "[VLM failed, no output]"
                else:
                    status = "skipped"
                    converted_content = "[image not found]"

            elif content_type == "image" and (not enable_image or not vision_model_func):
                # Image processing disabled or no VLM
                img_path = item.get("img_path", "")
                original_content = img_path
                status = "skipped"
                converted_content = "[image processing disabled]"

            else:
                # Unknown or unhandled type
                original_content = str(item)[:200]
                status = "skipped"
                converted_content = f"[unhandled type: {content_type}]"

            # Collect block mapping if storage is provided
            if block_mapping_storage is not None:
                block_mappings.append(BlockMapping(
                    block_index=block_idx,
                    page_idx=page_idx,
                    block_type=content_type,
                    bbox=bbox,
                    original_content=original_content,
                    converted_content=converted_content,
                    processing_time=processing_time,
                    status=status,
                    error_message=error_message,
                ))

        content = "\n\n".join(text_parts)

        if not content.strip():
            logger.warning(f"No text content extracted from {file_path.name}")

        total_elapsed = time.time() - parse_start_time

        # Save block mappings if storage is provided
        if block_mapping_storage is not None and block_mappings:
            doc_mappings = DocumentBlockMappings(
                doc_id=doc_id,
                file_path=str(file_path),
                parser=parser,
                total_blocks=len(content_list),
                processed_blocks=len([b for b in block_mappings if b.status == "success"]),
                total_processing_time=total_elapsed,
                created_at=datetime.now().isoformat(),
                blocks=block_mappings,
                text_blocks=stats["text_blocks"],
                image_blocks=stats["image_blocks"],
                table_blocks=stats["table_blocks"],
                equation_blocks=stats["equation_blocks"],
                failed_blocks=stats["failed_blocks"],
            )
            await block_mapping_storage.save_mappings(doc_mappings)
            logger.info(
                f"[RAG-Anything] Block mappings saved: {len(block_mappings)} blocks "
                f"(text:{stats['text_blocks']}, image:{stats['image_blocks']}, "
                f"table:{stats['table_blocks']}, equation:{stats['equation_blocks']}, "
                f"failed:{stats['failed_blocks']})"
            )

        logger.info(
            f"[RAG-Anything] Completed {file_path.name}: {len(content)} chars, "
            f"{image_count} images processed, total {total_elapsed:.1f}s"
        )
        return content if content.strip() else ""

    except Exception as e:
        logger.error(f"RAG-Anything extraction error for {file_path.name}: {e}")
        raise


async def extract_text_with_raganything(
    file_path: Path,
    vision_model_func: Optional[Callable] = None,
    enable_image: bool = True,
    enable_table: bool = True,
    enable_equation: bool = True,
    parse_method: str = "auto",
) -> str:
    """Extract text from a file using RAG-Anything without inserting to LightRAG.

    This is useful when you want to get the processed text content
    without automatically inserting it into LightRAG.

    Args:
        file_path: Path to the file to process
        vision_model_func: Function for processing images with vision model
        enable_image: Enable image processing
        enable_table: Enable table processing
        enable_equation: Enable equation processing
        parse_method: Parsing method ('auto', 'ocr', 'txt')

    Returns:
        str: Extracted and processed text content

    Raises:
        ImportError: If raganything is not installed
    """
    if not _is_raganything_available():
        raise ImportError(
            "RAG-Anything is not installed. "
            "Install it with: pip install raganything"
        )

    try:
        # Import MinerU parser directly for text extraction only
        from magic_pdf.data.data_reader_writer import FileBasedDataReader
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.data.dataset import PymuDocDataset

        # Read the file
        reader = FileBasedDataReader("")
        file_bytes = reader.read(str(file_path))

        # Create dataset and analyze
        dataset = PymuDocDataset(file_bytes)

        # Get markdown content
        if parse_method == "ocr":
            infer_result = dataset.apply(doc_analyze, ocr=True)
        elif parse_method == "txt":
            infer_result = dataset.apply(doc_analyze, ocr=False)
        else:  # auto
            infer_result = dataset.apply(doc_analyze)

        # Extract text content
        content_list = infer_result.pipe_txt_mode.get_markdown()
        content = "\n\n".join(content_list) if isinstance(content_list, list) else content_list

        # Process images if enabled and vision model is available
        if enable_image and vision_model_func:
            images = infer_result.pipe_txt_mode.get_images()
            if images:
                image_descriptions = []
                for i, img_data in enumerate(images):
                    try:
                        # Convert image to base64 if needed
                        if isinstance(img_data, bytes):
                            img_b64 = base64.b64encode(img_data).decode("utf-8")
                        else:
                            img_b64 = img_data

                        description = await vision_model_func(
                            "Describe this image in detail, focusing on any text, "
                            "diagrams, charts, or important visual information.",
                            img_b64,
                        )
                        image_descriptions.append(f"[Image {i+1}]: {description}")
                    except Exception as e:
                        logger.warning(f"Failed to process image {i+1}: {e}")

                if image_descriptions:
                    content += "\n\n## Image Descriptions\n\n" + "\n\n".join(image_descriptions)

        return content

    except ImportError:
        # Fallback: try using raganything's built-in parser
        logger.warning("MinerU not available, using fallback method")

        from raganything.parser import parse_document

        result = await asyncio.to_thread(
            parse_document,
            str(file_path),
            parse_method=parse_method,
        )

        return result.get("content", "")

    except Exception as e:
        logger.error(f"Text extraction error for {file_path.name}: {e}")
        raise


def encode_image_to_base64(file_path: Path) -> str:
    """Encode an image file to base64.

    Args:
        file_path: Path to the image file

    Returns:
        str: Base64 encoded image string
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
