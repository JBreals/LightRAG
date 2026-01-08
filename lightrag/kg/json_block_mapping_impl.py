"""
JSON-based storage for document block mappings.

This module stores the mapping between original document blocks (from MinerU/Docling)
and their converted content for debugging and visualization purposes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import os

from lightrag.utils import load_json, logger, write_json


@dataclass
class BlockMapping:
    """Represents a single block mapping from document parsing."""

    block_index: int
    page_idx: int
    block_type: str  # text, image, table, equation
    bbox: list[float]  # [x0, y0, x1, y1] normalized 0-1000
    original_content: str  # text preview, image path, etc.
    converted_content: str  # what was added to output
    processing_time: Optional[float] = None  # seconds, for VLM processing
    status: str = "success"  # success, failed, skipped
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "page_idx": self.page_idx,
            "block_type": self.block_type,
            "bbox": self.bbox,
            "original_content": self.original_content,
            "converted_content": self.converted_content,
            "processing_time": self.processing_time,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockMapping":
        return cls(
            block_index=data.get("block_index", 0),
            page_idx=data.get("page_idx", 0),
            block_type=data.get("block_type", "unknown"),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            original_content=data.get("original_content", ""),
            converted_content=data.get("converted_content", ""),
            processing_time=data.get("processing_time"),
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
        )


@dataclass
class DocumentBlockMappings:
    """Represents all block mappings for a single document."""

    doc_id: str
    file_path: str
    parser: str  # mineru, docling
    total_blocks: int
    processed_blocks: int
    total_processing_time: float
    created_at: str
    blocks: list[BlockMapping] = field(default_factory=list)

    # Summary statistics
    text_blocks: int = 0
    image_blocks: int = 0
    table_blocks: int = 0
    equation_blocks: int = 0
    failed_blocks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "parser": self.parser,
            "total_blocks": self.total_blocks,
            "processed_blocks": self.processed_blocks,
            "total_processing_time": self.total_processing_time,
            "created_at": self.created_at,
            "text_blocks": self.text_blocks,
            "image_blocks": self.image_blocks,
            "table_blocks": self.table_blocks,
            "equation_blocks": self.equation_blocks,
            "failed_blocks": self.failed_blocks,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentBlockMappings":
        blocks = [BlockMapping.from_dict(b) for b in data.get("blocks", [])]
        return cls(
            doc_id=data.get("doc_id", ""),
            file_path=data.get("file_path", ""),
            parser=data.get("parser", "unknown"),
            total_blocks=data.get("total_blocks", 0),
            processed_blocks=data.get("processed_blocks", 0),
            total_processing_time=data.get("total_processing_time", 0.0),
            created_at=data.get("created_at", ""),
            text_blocks=data.get("text_blocks", 0),
            image_blocks=data.get("image_blocks", 0),
            table_blocks=data.get("table_blocks", 0),
            equation_blocks=data.get("equation_blocks", 0),
            failed_blocks=data.get("failed_blocks", 0),
            blocks=blocks,
        )


@dataclass
class JsonBlockMappingStorage:
    """JSON-based storage for document block mappings."""

    working_dir: str
    workspace: str = ""

    def __post_init__(self):
        if self.workspace:
            workspace_dir = os.path.join(self.working_dir, self.workspace)
        else:
            workspace_dir = self.working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, "block_mappings.json")
        self._data: dict[str, dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize storage by loading existing data."""
        if self._initialized:
            return

        loaded_data = load_json(self._file_name) or {}
        self._data = loaded_data
        self._initialized = True
        logger.info(
            f"[{self.workspace}] Block mapping storage loaded with {len(self._data)} documents"
        )

    async def _save(self):
        """Persist data to disk."""
        write_json(self._data, self._file_name)

    async def save_mappings(self, mappings: DocumentBlockMappings) -> None:
        """Save block mappings for a document.

        Args:
            mappings: The document block mappings to save
        """
        if not self._initialized:
            await self.initialize()

        self._data[mappings.doc_id] = mappings.to_dict()
        await self._save()
        logger.debug(
            f"[{self.workspace}] Saved {len(mappings.blocks)} block mappings for {mappings.file_path}"
        )

    async def get_by_doc_id(self, doc_id: str) -> Optional[DocumentBlockMappings]:
        """Get block mappings for a specific document.

        Args:
            doc_id: The document ID

        Returns:
            DocumentBlockMappings if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        data = self._data.get(doc_id)
        if data:
            return DocumentBlockMappings.from_dict(data)
        return None

    async def get_by_file_path(self, file_path: str) -> Optional[DocumentBlockMappings]:
        """Get block mappings by file path.

        Supports both exact path matching and filename-only matching.
        If file_path is just a filename (no directory separator), it will
        match against the basename of stored paths.

        Args:
            file_path: The file path or filename to search for

        Returns:
            DocumentBlockMappings if found, None otherwise
        """
        import os

        if not self._initialized:
            await self.initialize()

        # Check if file_path is just a filename (no directory)
        is_filename_only = os.sep not in file_path and "/" not in file_path

        for doc_id, data in self._data.items():
            stored_path = data.get("file_path", "")

            # Exact match
            if stored_path == file_path:
                return DocumentBlockMappings.from_dict(data)

            # Filename-only match (compare basenames)
            if is_filename_only and os.path.basename(stored_path) == file_path:
                return DocumentBlockMappings.from_dict(data)

            # Also try matching the query against stored basename
            if os.path.basename(stored_path) == os.path.basename(file_path):
                return DocumentBlockMappings.from_dict(data)

        return None

    async def get_all_summaries(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get paginated list of document summaries (without full block details).

        Args:
            page: Page number (1-based)
            page_size: Number of items per page

        Returns:
            Tuple of (list of summary dicts, total count)
        """
        if not self._initialized:
            await self.initialize()

        # Create summaries without full block details
        summaries = []
        for doc_id, data in self._data.items():
            summary = {
                "doc_id": doc_id,
                "file_path": data.get("file_path", ""),
                "parser": data.get("parser", ""),
                "total_blocks": data.get("total_blocks", 0),
                "processed_blocks": data.get("processed_blocks", 0),
                "total_processing_time": data.get("total_processing_time", 0.0),
                "created_at": data.get("created_at", ""),
                "text_blocks": data.get("text_blocks", 0),
                "image_blocks": data.get("image_blocks", 0),
                "table_blocks": data.get("table_blocks", 0),
                "equation_blocks": data.get("equation_blocks", 0),
                "failed_blocks": data.get("failed_blocks", 0),
            }
            summaries.append(summary)

        # Sort by created_at descending
        summaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        total_count = len(summaries)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = summaries[start_idx:end_idx]

        return paginated, total_count

    async def delete(self, doc_id: str) -> bool:
        """Delete block mappings for a document.

        Args:
            doc_id: The document ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            await self.initialize()

        if doc_id in self._data:
            del self._data[doc_id]
            await self._save()
            return True
        return False

    async def drop(self) -> dict[str, str]:
        """Drop all block mapping data.

        Returns:
            Status dict with result
        """
        try:
            self._data.clear()
            await self._save()
            logger.info(f"[{self.workspace}] Block mapping storage dropped")
            return {"status": "success", "message": "Block mapping data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping block mappings: {e}")
            return {"status": "error", "message": str(e)}
