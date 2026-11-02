from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import PROMPTS_DIR


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    description: str
    system: str
    user: str

    def render_user(self, context: str, question: str) -> str:
        return self.user.replace("{context}", context).replace("{question}", question)


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    path = Path(prompts_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found in {prompts_dir}")
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("name", "version", "system", "user"):
        if key not in d:
            raise ValueError(f"{path.name} is missing '{key}'")
    if "{context}" not in d["user"] or "{question}" not in d["user"]:
        raise ValueError(f"{path.name} user template must contain {{context}} and {{question}}")
    return Prompt(d["name"], int(d["version"]), d.get("description", ""), d["system"].strip(), d["user"].strip())


def list_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(prompts_dir).glob("*.yaml"))
