"""Knowledge-base build pipeline for the DAC-3D assistant."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import defusedxml.ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AppConfig  # noqa: E402

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s+|\n{2,}")
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.\d+(?:\.\d+)*|[一二三四五六七八九十]+[、.])\s*[:：.]?\s*(.+)$"
)
DOCX_NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
SUPPORTED_DOCUMENT_FORMATS = frozenset({"text", "pdf", "docx"})


@dataclass(slots=True)
class SourceDocument:
    """Raw document loaded from the knowledge base directory."""

    source: str
    title: str
    document_type: str
    text: str
    path: str


class EmbeddingBackend:
    """Simple interface used by the build and retrieval steps."""

    name: str = "embedding-backend"
    dimensions: int = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic fallback embedder for offline environments."""

    def __init__(self, dimensions: int = 384) -> None:
        self.name = "hashing"
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.dimensions) for text in texts]


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str, dimensions: int, *, local_files_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = "sentence-transformers"
        self.dimensions = dimensions
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


def normalize_text(text: str) -> str:
    """Normalize line endings and trim the document."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def tokenize(text: str) -> list[str]:
    """Tokenize English and Chinese text for lexical fallback scoring."""
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(raw_token)
        if raw_token.isascii():
            continue
        if len(raw_token) <= 4:
            tokens.extend(list(raw_token))
        if len(raw_token) > 1:
            tokens.extend(raw_token[index : index + 2] for index in range(len(raw_token) - 1))
    return tokens


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two dense vectors."""
    if not left or not right:
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def create_embedding_backend(config: AppConfig, preferred_backend: str | None = None) -> EmbeddingBackend:
    """Create the configured embedding backend with an offline fallback."""
    backend_name = (preferred_backend or "").strip().lower()
    if backend_name == "hashing":
        return HashingEmbeddingBackend(config.embedding_dimensions)

    model_path = Path(config.embedding_model_name)
    if not config.embedding_download_allowed and not model_path.exists():
        return HashingEmbeddingBackend(config.embedding_dimensions)

    try:
        return SentenceTransformerEmbeddingBackend(
            model_name=config.embedding_model_name,
            dimensions=config.embedding_dimensions,
            local_files_only=not config.embedding_download_allowed,
        )
    except Exception:
        return HashingEmbeddingBackend(config.embedding_dimensions)


def infer_document_type(path: Path) -> str:
    """Infer a lightweight document type from the filename."""
    stem = path.stem.lower()
    if any(keyword in stem for keyword in ("defect", "interpret", "缺陷", "结果")):
        return "defect"
    if any(keyword in stem for keyword in ("trouble", "guide", "reflect", "建议", "故障")):
        return "guidance"
    if any(keyword in stem for keyword in ("workflow", "status", "流程", "状态")):
        return "workflow"
    if any(keyword in stem for keyword in ("parameter", "manual", "参数", "手册")):
        return "manual"
    return "reference"


def _looks_like_text(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    suspicious_bytes = 0
    for value in sample:
        if value in (9, 10, 13):
            continue
        if value < 32:
            suspicious_bytes += 1
    return (suspicious_bytes / max(len(sample), 1)) < 0.02


def _inspect_zip_container(path: Path) -> str | None:
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return None

    if "word/document.xml" in names:
        return "docx"
    return None


def detect_document_format(path: Path) -> str | None:
    """Detect the actual document format from content before using the suffix as fallback."""
    if not path.is_file():
        return None

    header = path.read_bytes()[:4096]
    if not header:
        return None
    if header.startswith(b"%PDF-"):
        return "pdf"
    if header.startswith(b"PK\x03\x04"):
        detected_zip_format = _inspect_zip_container(path)
        if detected_zip_format is not None:
            return detected_zip_format
    if header.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "doc"
    if _looks_like_text(header):
        return "text"

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix == ".doc":
        return "doc"
    if suffix in {".md", ".txt"}:
        return "text"
    return None


def describe_document_format(path: Path) -> str:
    detected_format = detect_document_format(path)
    if detected_format == "doc":
        return "Legacy Microsoft Word .doc files are detected by content but are not supported. Convert them to .docx, .pdf, .md, or .txt first."
    if detected_format in SUPPORTED_DOCUMENT_FORMATS:
        return detected_format
    return "Unsupported or unknown document format."


def _read_text_with_fallbacks(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    for encoding in encodings:
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
        styles_xml = archive.read("word/styles.xml") if "word/styles.xml" in archive.namelist() else b""

    style_names: dict[str, str] = {}
    if styles_xml:
        styles_root = ET.fromstring(styles_xml)
        for style in styles_root.findall(".//w:style", DOCX_NAMESPACES):
            style_id = style.attrib.get(f"{{{DOCX_NAMESPACES['w']}}}styleId", "")
            style_name = style.find("w:name", DOCX_NAMESPACES)
            if style_id:
                style_names[style_id] = style_name.attrib.get(
                    f"{{{DOCX_NAMESPACES['w']}}}val",
                    "",
                ) if style_name is not None else ""

    document_root = ET.fromstring(document_xml)
    lines: list[str] = []
    for paragraph in document_root.findall(".//w:body/w:p", DOCX_NAMESPACES):
        texts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{DOCX_NAMESPACES['w']}}}t" and node.text:
                texts.append(node.text)
            elif node.tag in {
                f"{{{DOCX_NAMESPACES['w']}}}tab",
                f"{{{DOCX_NAMESPACES['w']}}}br",
                f"{{{DOCX_NAMESPACES['w']}}}cr",
            }:
                texts.append("\n")
        paragraph_text = normalize_text("".join(texts))
        if not paragraph_text:
            continue

        style_node = paragraph.find("w:pPr/w:pStyle", DOCX_NAMESPACES)
        style_id = (
            style_node.attrib.get(f"{{{DOCX_NAMESPACES['w']}}}val", "")
            if style_node is not None
            else ""
        )
        style_name = style_names.get(style_id, style_id).lower()
        if style_name.startswith("heading") or "heading" in style_name or "标题" in style_name:
            lines.append(f"# {paragraph_text}")
            continue
        lines.append(paragraph_text)

    return normalize_text("\n".join(lines))


def _read_document_text(path: Path) -> str:
    document_format = detect_document_format(path)
    if document_format == "text":
        return _read_text_with_fallbacks(path)

    if document_format == "pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PDF support requires the optional dependency 'pypdf'."
            ) from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))

    if document_format == "docx":
        return _read_docx_text(path)

    raise ValueError(describe_document_format(path))


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
        return stripped[:120]
    return path.stem.replace("_", " ").title()


def _iter_document_paths(documents_dir: Path, patterns: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(documents_dir.glob(pattern)):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    paths.sort()
    return paths


def load_documents(
    documents_dir: Path,
    patterns: Sequence[str] | None = None,
) -> list[SourceDocument]:
    """Load supported knowledge-base documents from disk using content-based format detection."""
    documents: list[SourceDocument] = []
    for path in _iter_document_paths(documents_dir, patterns or ("**/*",)):
        detected_format = detect_document_format(path)
        if detected_format not in SUPPORTED_DOCUMENT_FORMATS:
            continue
        text = _read_document_text(path)
        if not text:
            continue
        title = _extract_title(text, path)
        documents.append(
            SourceDocument(
                source=path.name,
                title=title,
                document_type=infer_document_type(path),
                text=text,
                path=str(path),
            )
        )
    return documents


def _parse_heading_line(line: str, default_title: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return stripped.lstrip("# ").strip() or default_title

    match = NUMBERED_HEADING_PATTERN.match(stripped)
    if not match:
        return None
    heading_text = match.group(1).strip()
    if not heading_text or len(heading_text) > 80:
        return None
    return heading_text


def _iter_sections(document: SourceDocument) -> Iterable[tuple[str, str]]:
    current_section = document.title
    buffer: list[str] = []
    for line in document.text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = _parse_heading_line(stripped, document.title)
        if heading:
            if buffer:
                yield current_section, " ".join(buffer).strip()
                buffer.clear()
            current_section = heading
            continue
        buffer.append(stripped)
    if buffer:
        yield current_section, " ".join(buffer).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return sentences or [text.strip()]


def _token_budget(text: str) -> int:
    budget = len(tokenize(text))
    return budget if budget > 0 else max(1, len(text) // 4)


def _window_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return [text]
    windows: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            continue
        windows.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return windows


def _chunk_section_text(section_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = _split_sentences(section_text)
    chunks: list[str] = []
    overlap_sentence_count = 1 if chunk_overlap > 0 else 0
    current_sentences: list[str] = []
    current_budget = 0

    for sentence in sentences:
        sentence_budget = _token_budget(sentence)
        if sentence_budget > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_budget = 0
            chunks.extend(_window_long_text(sentence, chunk_size, chunk_overlap))
            continue

        if current_sentences and current_budget + sentence_budget > chunk_size:
            chunks.append(" ".join(current_sentences).strip())
            overlap_sentences = current_sentences[-overlap_sentence_count:] if overlap_sentence_count else []
            current_sentences = list(overlap_sentences)
            current_budget = sum(_token_budget(item) for item in current_sentences)

        current_sentences.append(sentence)
        current_budget += sentence_budget

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    """Split documents into heading-aware chunks with metadata."""
    chunks: list[dict[str, object]] = []
    chunk_id = 0
    for document in documents:
        for section, section_text in _iter_sections(document):
            for chunk_text in _chunk_section_text(section_text, chunk_size, chunk_overlap):
                chunk_tokens = tokenize(chunk_text)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "source": document.source,
                        "title": document.title,
                        "section": section,
                        "document_type": document.document_type,
                        "path": document.path,
                        "token_counts": dict(Counter(chunk_tokens)),
                        "metadata": {
                            "chunk_id": chunk_id,
                            "source": document.source,
                            "title": document.title,
                            "section": section,
                            "document_type": document.document_type,
                            "path": document.path,
                        },
                    }
                )
                chunk_id += 1
    return chunks


def embed_chunks(
    chunks: list[dict[str, object]],
    config: AppConfig,
    *,
    preferred_backend: str | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Embed chunks and attach dense vectors to each record."""
    backend = create_embedding_backend(config, preferred_backend=preferred_backend)
    embeddings = backend.embed_texts([str(chunk["text"]) for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks, backend.name


def _persist_with_chroma(chunks: list[dict[str, object]], config: AppConfig) -> bool:
    try:
        import chromadb
    except Exception:
        return False

    config.vector_store_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.vector_store_path))
    from chromadb.errors import NotFoundError

    with suppress(NotFoundError):
        client.delete_collection(config.vector_store_collection)
    collection = client.get_or_create_collection(
        name=config.vector_store_collection,
        metadata={"hnsw:space": "cosine"},
    )
    add_method = getattr(collection, "upsert", None) or getattr(collection, "add")
    add_method(
        ids=[str(chunk["chunk_id"]) for chunk in chunks],
        documents=[str(chunk["text"]) for chunk in chunks],
        metadatas=[dict(chunk["metadata"]) for chunk in chunks],
        embeddings=[list(chunk["embedding"]) for chunk in chunks],
    )
    return True


def load_vector_store_manifest(config: AppConfig) -> dict[str, Any]:
    """Load the persisted vector-store manifest if it exists."""
    if not config.vector_store_manifest_path.exists():
        return {}
    return json.loads(config.vector_store_manifest_path.read_text(encoding="utf-8"))


def load_build_history(config: AppConfig) -> list[dict[str, Any]]:
    """Load persisted build history entries."""
    if not config.vector_store_history_path.exists():
        return []
    return list(json.loads(config.vector_store_history_path.read_text(encoding="utf-8")))


def _append_build_history(
    config: AppConfig,
    manifest_payload: dict[str, Any],
    *,
    build_context: dict[str, Any] | None,
) -> None:
    history = load_build_history(config)
    documents = sorted(
        {
            str(chunk.get("source", ""))
            for chunk in manifest_payload.get("chunks", [])
            if str(chunk.get("source", ""))
        }
    )
    entry = {
        "generated_at": manifest_payload.get("generated_at"),
        "chunk_count": manifest_payload.get("chunk_count", 0),
        "document_count": len(documents),
        "documents": documents,
        "storage_backend": manifest_payload.get("storage_backend", "manifest"),
        "embedding_backend": manifest_payload.get("embedding_backend", "unknown"),
        "trigger": (build_context or {}).get("trigger", "manual_build"),
        "uploaded_files": list((build_context or {}).get("uploaded_files", [])),
    }
    history.insert(0, entry)
    config.vector_store_history_path.write_text(
        json.dumps(history[:30], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def summarize_knowledge_base(config: AppConfig) -> dict[str, Any]:
    """Return a compact summary of the current local knowledge base."""
    manifest = load_vector_store_manifest(config)
    document_paths = [
        path
        for path in _iter_document_paths(config.documents_dir, config.document_globs)
        if detect_document_format(path) in SUPPORTED_DOCUMENT_FORMATS
    ]
    history = load_build_history(config)
    return {
        "documents": [path.name for path in document_paths],
        "document_count": len(document_paths),
        "chunk_count": int(manifest.get("chunk_count", 0) or 0),
        "latest_build_at": manifest.get("generated_at"),
        "storage_backend": manifest.get("storage_backend", config.vector_store_type),
        "embedding_backend": manifest.get("embedding_backend", "unknown"),
        "history": history,
    }


def persist_vector_store(
    chunks: list[dict[str, object]],
    config: AppConfig,
    embedding_backend: str,
    *,
    build_context: dict[str, Any] | None = None,
) -> Path:
    """Persist chunk metadata and vectors to the configured store."""
    config.vector_store_path.mkdir(parents=True, exist_ok=True)
    storage_backend = "manifest"
    if config.vector_store_type.lower() == "chroma" and _persist_with_chroma(chunks, config):
        storage_backend = "chroma"

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "vector_store_type": config.vector_store_type,
        "storage_backend": storage_backend,
        "embedding_backend": embedding_backend,
        "embedding_model_name": config.embedding_model_name,
        "embedding_dimensions": len(chunks[0]["embedding"]) if chunks else config.embedding_dimensions,
        "chunk_count": len(chunks),
        "generated_at": generated_at,
        "chunks": chunks,
    }
    config.vector_store_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _append_build_history(config, payload, build_context=build_context)
    return config.vector_store_path


def build_knowledge_base(
    config: AppConfig | None = None,
    *,
    build_context: dict[str, Any] | None = None,
) -> Path:
    """Prepare document chunks and persist the local vector store."""
    active_config = config or AppConfig.from_env(ROOT_DIR)
    active_config.ensure_directories()
    documents = load_documents(active_config.documents_dir, active_config.document_globs)
    if not documents:
        raise FileNotFoundError(
            f"No knowledge-base documents found in {active_config.documents_dir}"
        )

    chunks = chunk_documents(
        documents,
        chunk_size=active_config.chunk_size,
        chunk_overlap=active_config.chunk_overlap,
    )
    embedded_chunks, backend_name = embed_chunks(chunks, active_config)
    return persist_vector_store(
        embedded_chunks,
        active_config,
        backend_name,
        build_context=build_context,
    )


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def main() -> None:
    """CLI entry point used by the repository build command."""
    config = AppConfig.from_env(ROOT_DIR)
    vector_store_path = build_knowledge_base(config)
    print(f"Knowledge base built at: {vector_store_path}")


if __name__ == "__main__":
    main()
