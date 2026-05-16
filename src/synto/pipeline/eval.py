"""Offline-first structural eval harness for Phase 1A."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from ..config import Config
from ..readers import VaultReader
from ..vault import extract_wikilinks, parse_note


@dataclass(frozen=True)
class EvalQuery:
    id: str
    question: str
    expected_concepts: tuple[str, ...]
    expected_contains: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    article_coverage: float | None
    term_recall: float | None
    citation_coverage: float | None
    index_json_validity: float | None
    wikilink_resolution: float | None
    harmonic_mean: float | None
    details: dict[str, object]


def run_offline(config: Config, queries_toml: Path | None = None) -> EvalResult:
    queries = load_queries(queries_toml)
    reader = VaultReader(config.vault)

    article_coverage, article_details = _article_coverage(reader, queries)
    citation_coverage, citation_details = _citation_coverage(reader)
    index_validity, index_details = _index_json_validity(config)
    wikilink_resolution, wikilink_details = _wikilink_resolution(reader)
    term_recall = None
    harmonic_mean = _harmonic_mean_skip_none(
        [article_coverage, term_recall, citation_coverage, index_validity, wikilink_resolution]
    )

    return EvalResult(
        article_coverage=article_coverage,
        term_recall=term_recall,
        citation_coverage=citation_coverage,
        index_json_validity=index_validity,
        wikilink_resolution=wikilink_resolution,
        harmonic_mean=harmonic_mean,
        details={
            "queries_evaluated": len(queries),
            "metrics_skipped": ["term_recall"],
            "article_coverage": article_details,
            "citation_coverage": citation_details,
            "index_json_validity": index_details,
            "wikilink_resolution": wikilink_details,
        },
    )


def run_live(config: Config, queries_toml: Path | None = None) -> EvalResult:
    raise SystemExit(
        "synto eval --live is not implemented in Phase 1A. Use offline mode without --live."
    )


def render_json(result: EvalResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True)


def render_text(result: EvalResult) -> str:
    lines = [
        "synto eval",
        "",
        f"Article coverage: {format_metric(result.article_coverage)}",
        f"Term recall: {format_metric(result.term_recall)}",
        f"Citation coverage: {format_metric(result.citation_coverage)}",
        f"INDEX.json validity: {format_metric(result.index_json_validity)}",
        f"Wikilink resolution: {format_metric(result.wikilink_resolution)}",
        f"Harmonic mean: {format_metric(result.harmonic_mean)}",
    ]
    return "\n".join(lines)


def load_queries(queries_toml: Path | None = None) -> list[EvalQuery]:
    path = queries_toml or (
        Path(__file__).resolve().parents[3] / "tests" / "eval" / "queries_default.toml"
    )
    if not path.exists():
        return []
    with open(path, "rb") as f:
        payload = tomllib.load(f)
    queries: list[EvalQuery] = []
    for entry in payload.get("query", []):
        queries.append(
            EvalQuery(
                id=str(entry["id"]),
                question=str(entry["question"]),
                expected_concepts=tuple(str(v) for v in entry.get("expected_concepts", [])),
                expected_contains=tuple(str(v) for v in entry.get("expected_contains", [])),
            )
        )
    return queries


def _article_coverage(
    reader: VaultReader, queries: list[EvalQuery]
) -> tuple[float, dict[str, object]]:
    expected_total = sum(len(query.expected_concepts) for query in queries)
    if expected_total == 0:
        return 1.0, {"matched_concepts": 0, "expected_concepts": 0}

    matched = 0
    missing: dict[str, list[str]] = {}
    for query in queries:
        for concept in query.expected_concepts:
            if reader.find_concept(concept) is not None or _has_article(reader, concept):
                matched += 1
            else:
                missing.setdefault(query.id, []).append(concept)
    return matched / expected_total, {
        "matched_concepts": matched,
        "expected_concepts": expected_total,
        "missing": missing,
    }


def _citation_coverage(reader: VaultReader) -> tuple[float | None, dict[str, object]]:
    refs = reader.list_articles()
    if not refs:
        return None, {"with_sources": 0, "articles": 0}

    with_sources = 0
    for ref in refs:
        article = reader.read_article(ref.id)
        sources = article.frontmatter.get("sources", [])
        if isinstance(sources, list) and len(sources) > 0:
            with_sources += 1
    return with_sources / len(refs), {"with_sources": with_sources, "articles": len(refs)}


def _index_json_validity(config: Config) -> tuple[float, dict[str, object]]:
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return 0.0, {"reason": "missing"}

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0, {"reason": "invalid_json"}

    try:
        import jsonschema

        schema_path = files("synto.schemas").joinpath("index-v1.json")
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        return 0.0, {"reason": "schema_invalid", "detail": exc.message}
    return 1.0, {"reason": "ok"}


def _wikilink_resolution(reader: VaultReader) -> tuple[float | None, dict[str, object]]:
    refs = reader.list_articles()
    total = 0
    resolved = 0
    broken: list[dict[str, str]] = []
    article_names = {ref.name.casefold() for ref in refs}
    article_stems = {Path(ref.path).stem.casefold() for ref in refs}

    for ref in refs:
        article = reader.read_article(ref.id)
        for link in extract_wikilinks(article.body):
            total += 1
            link_cf = link.casefold()
            if link_cf in article_names or link_cf in article_stems:
                resolved += 1
                continue
            if _has_source_page(reader, link):
                resolved += 1
                continue
            if reader.find_concept(link) is not None:
                resolved += 1
                continue
            broken.append({"article": ref.path, "link": link})

    if total == 0:
        return 1.0, {"resolved": 0, "total": 0, "broken": []}
    return resolved / total, {"resolved": resolved, "total": total, "broken": broken}


def _has_article(reader: VaultReader, concept: str) -> bool:
    try:
        reader.read_article(concept)
    except Exception:
        return False
    return True


def _has_source_page(reader: VaultReader, link: str) -> bool:
    link = link.strip()
    source_name = link.split("/", 1)[1].strip() if link.casefold().startswith("sources/") else link
    if not source_name:
        return False

    vault_root = reader.vault_root
    exact = vault_root / "wiki" / "sources" / f"{source_name}.md"
    if exact.exists():
        return True

    for path in (vault_root / "wiki" / "sources").glob("*.md"):
        if path.stem.casefold() == source_name.casefold():
            return True
        try:
            meta, _ = parse_note(path)
        except Exception:
            continue
        title = str(meta.get("title", path.stem))
        if title.casefold() == source_name.casefold():
            return True
    return False


def _harmonic_mean_skip_none(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    if any(value <= 0 for value in defined):
        return 0.0
    return len(defined) / sum(1.0 / value for value in defined)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
