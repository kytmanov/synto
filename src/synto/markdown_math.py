from __future__ import annotations

import re

_INLINE_MATH_RE = r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$(?!\$)"
_BASE_MASK_PARTS = [
    r"```[\s\S]*?```",
    r"`[^`]+`",
    r"\$\$[\s\S]*?\$\$",
    _INLINE_MATH_RE,
    r"\\\([\s\S]*?\\\)",
    r"!\[[^\]]*\]\([^)]*\)",
    r"\[[^\]\n]+\]\([^)]*\)",
]
_OBSIDIAN_EMBED_RE = r"!\[\[[\s\S]*?\]\]"
_WIKILINK_RE = r"\[\[[^\]]+\]\]"
_DISPLAY_MATH_RE = re.compile(r"\\{1,2}\[(.*?)\\{1,2}\]", re.DOTALL)
_BARE_LATEX_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<body>\\{1,2}[A-Za-z{_].*)$")
_LATEX_COMMAND_RE = re.compile(
    r"^\\(?:begin|end|frac|sqrt|sum|prod|int|lim|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|phi|psi|omega|sin|cos|tan|log|ln|exp|cdot|times|leq|geq|neq|approx|left|right|text|mathrm|mathbf|mathit|operatorname)\b"
)
_MATH_SIGNAL_RE = re.compile(r"[{}_^=]|\\(?:[A-Za-z]+|[,;! ])")


def mask_markdown_regions(
    content: str, *, mask_wikilinks: bool = True, mask_embeds: bool = True
) -> tuple[str, list[tuple[str, str]]]:
    """Protect markdown regions that should not be rewritten."""
    replacements: list[tuple[str, str]] = []
    parts = list(_BASE_MASK_PARTS)
    if mask_embeds:
        parts.append(_OBSIDIAN_EMBED_RE)
    if mask_wikilinks:
        parts.append(_WIKILINK_RE)
    pattern = re.compile("|".join(parts), re.MULTILINE)

    def replace(match: re.Match[str]) -> str:
        token = f"__SYNTO_MARKDOWN_MASK_{len(replacements)}__"
        replacements.append((token, match.group(0)))
        return token

    return pattern.sub(replace, content), replacements


def restore_markdown_regions(content: str, replacements: list[tuple[str, str]]) -> str:
    for token, original in replacements:
        content = content.replace(token, original)
    return content


def _display_math_repl(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not inner:
        return match.group(0)
    return f"$$\n{inner}\n$$"


def _looks_like_bare_latex_line(stripped: str) -> bool:
    normalized = stripped.lstrip("\\")
    if stripped.startswith(("\\[", "\\\\[", "\\(", "\\\\(")):
        return False
    if stripped.startswith(("#", ">", "- ", "* ", "+ ", "|")):
        return False
    if re.match(r"\d+\.\s", stripped):
        return False
    if _LATEX_COMMAND_RE.match(stripped):
        return True
    return bool(_MATH_SIGNAL_RE.search(normalized))


def sanitize_obsidian_math(content: str) -> str:
    """Normalize LLM LaTeX output to Obsidian-friendly math delimiters."""
    masked, replacements = mask_markdown_regions(content)
    masked = _DISPLAY_MATH_RE.sub(_display_math_repl, masked)

    lines: list[str] = []
    for line in masked.splitlines(keepends=True):
        line_ending = ""
        if line.endswith("\r\n"):
            line_ending = "\r\n"
            raw = line[:-2]
        elif line.endswith("\n"):
            line_ending = "\n"
            raw = line[:-1]
        else:
            raw = line

        match = _BARE_LATEX_LINE_RE.match(raw)
        if not match:
            lines.append(line)
            continue

        indent = match.group("indent")
        stripped = match.group("body").strip()
        if not _looks_like_bare_latex_line(stripped):
            lines.append(line)
            continue

        if stripped.startswith("\\\\") and not stripped.startswith("\\\\\\"):
            stripped = stripped[1:]

        ending = line_ending or "\n"
        lines.append(f"{indent}$$ {stripped} $${ending}")

    sanitized = "".join(lines)
    return restore_markdown_regions(sanitized, replacements)


def has_malformed_obsidian_math(content: str) -> bool:
    return sanitize_obsidian_math(content) != content
