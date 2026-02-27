import os
import threading
import logger
from agents import function_tool
from config import settings


log = logger.get(__name__)

# { name -> (skill_md_path, description) }
_skill_index: dict[str, tuple[str, str]] = {}
_skills_dir: str = ""
_dir_mtime: float = 0.0
_index_lock = threading.Lock()


def _read_frontmatter(path: str) -> dict:
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            if f.readline().strip() != "---":
                return result
            for line in f:
                line = line.rstrip("\n")
                if line == "---":
                    break
                if ":" in line:
                    k, _, v = line.partition(":")
                    result[k.strip()] = v.strip()
    except OSError:
        pass
    return result


def refresh_skill_catalog(skills_dir: str) -> None:
    global _skill_index, _skills_dir, _dir_mtime

    if not os.path.isdir(skills_dir):
        with _index_lock:
            if _skill_index:
                _skill_index = {}
        log.warning("skills dir not found: %s", skills_dir)
        return

    mtime = os.path.getmtime(skills_dir)
    with _index_lock:
        if skills_dir == _skills_dir and mtime == _dir_mtime:
            return

    index: dict[str, tuple[str, str]] = {}
    for entry in sorted(os.scandir(skills_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        skill_md = os.path.join(entry.path, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        meta = _read_frontmatter(skill_md)
        name = meta.get("name") or entry.name
        index[name] = (skill_md, meta.get("description", ""))

    with _index_lock:
        _skill_index = index
        _skills_dir = skills_dir
        _dir_mtime = mtime
    log.info("skill catalog refreshed: %d skill(s)", len(index))


def build_skill_catalog() -> str:
    with _index_lock:
        index = dict(_skill_index)
    if not index:
        return ""
    entries = [
        f"- **{name}**" + (f": {desc}" if desc else "")
        for name, (_, desc) in index.items()
    ]
    return (
        "# Available Skills\n\n"
        "Use the `get_skill` tool to retrieve full instructions for a skill before using it.\n\n"
        + "\n".join(entries)
    )


def build_soul_instructions(ctx, agent) -> str:
    base = settings.soul_prompt
    catalog = build_skill_catalog()
    if catalog:
        return f"{base}\n\n{catalog}"
    return base


@function_tool
def get_skill(skill_name: str) -> str:
    """
    Get the full usage guide for a specific skill by name.
    Call this before using any skill to retrieve detailed instructions.
    skill_name should match the name listed in the skill catalog (e.g. 'web-search').
    """
    key = skill_name.lower()
    with _index_lock:
        index = dict(_skill_index)
    for name, (path, _) in index.items():
        if name.lower() == key:
            log.info("getting skill: %s", skill_name)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return f"Skill '{skill_name}' not found. Check the skill catalog for available names."
