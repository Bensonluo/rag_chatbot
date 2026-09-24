"""Packaging contract: heavy optional-runtime deps stay out of base deps.

sentence-transformers drags the CUDA torch stack (6GB+) into every image
that installs the base package — a deploy on a full disk died on this
(2026-09-25). API-embedding deployments (EMBEDDING_PROVIDER=glm/openai)
must stay slim; on-prem embedding deps belong to the "local" extra only.
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"

# Anything whose install graph carries torch/CUDA or multi-GB wheels.
HEAVY_DEPS = ("sentence-transformers", "torch", "FlagEmbedding", "accelerate")


def _load() -> dict:  # type: ignore[type-arg]
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_heavy_optional_deps_are_not_base_dependencies() -> None:
    deps = [str(d) for d in _load()["project"]["dependencies"]]
    for heavy in HEAVY_DEPS:
        assert not any(heavy in d for d in deps), (
            f"{heavy} must live in [project.optional-dependencies] (extra 'local'), "
            f"not base dependencies — it bloats every Docker image by gigabytes"
        )


def test_local_extra_keeps_local_embedding_provider_installable() -> None:
    extras = _load()["project"]["optional-dependencies"]
    assert any("sentence-transformers" in str(d) for d in extras["local"]), (
        "pip install .[local] must still provide the on-prem embedding provider"
    )
