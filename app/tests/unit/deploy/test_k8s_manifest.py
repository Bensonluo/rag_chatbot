"""K8s manifest env contract: every injected var must be a real setting.

The manifest's own ConfigMap comment states the rule — keys injected
via env/envFrom ``MUST match Settings field names … otherwise the pod
will not start correctly or will silently fall back to single-instance
behavior''. A var naming a nonexistent field is silently ignored by
pydantic-settings: the deployment reads green while the app runs on
defaults (the QDRANT_URL drift did exactly this — pods booted with the
default vector-store address and a degraded retrieval stack while
/health stayed 200). These tests pin the rule the comment only states.
"""

from pathlib import Path
from typing import Any

import yaml

from app.config.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST_PATH = _REPO_ROOT / "deploy" / "k8s" / "deployment.yaml"

# Container-native vars that are legitimately not Settings fields
# (they configure the runtime, not the application).
_ALLOWED_NON_SETTINGS: frozenset[str] = frozenset()


def _manifest_docs() -> list[dict[str, Any]]:
    text = _MANIFEST_PATH.read_text(encoding="utf-8")
    return list(yaml.safe_load_all(text))


def _container_env_names() -> set[str]:
    names: set[str] = set()
    for doc in _manifest_docs():
        for container in _api_containers(doc):
            for entry in container.get("env") or []:
                if entry.get("name"):
                    names.add(entry["name"])
    return names


def _api_containers(doc: dict[str, Any]) -> list[dict[str, Any]]:
    if doc.get("kind") != "Deployment":
        return []
    template = doc["spec"]["template"]["spec"]
    return template.get("containers") or []


def _configmap_keys() -> set[str]:
    keys: set[str] = set()
    for doc in _manifest_docs():
        if doc.get("kind") == "ConfigMap":
            keys.update((doc.get("data") or {}).keys())
    return keys


class TestK8sManifestEnvContract:
    def test_every_container_env_names_a_real_setting(self):
        known = set(Settings.model_fields)
        unknown = _container_env_names() - known - _ALLOWED_NON_SETTINGS
        assert not unknown, (
            f"deployment.yaml injects env vars that are not Settings fields "
            f"(silently ignored — the pod runs on defaults): {sorted(unknown)}"
        )

    def test_every_configmap_key_names_a_real_setting(self):
        known = set(Settings.model_fields)
        unknown = _configmap_keys() - known - _ALLOWED_NON_SETTINGS
        assert not unknown, (
            f"rag-chatbot-config ConfigMap keys are not Settings fields "
            f"(injected as env but silently ignored): {sorted(unknown)}"
        )

    def test_vector_store_url_uses_the_settings_name(self):
        """The drift that motivated this contract: QDRANT_URL is not a
        field (VECTOR_DB_URL is), so pods silently booted against the
        default vector-store address with a degraded retrieval stack."""
        names = _container_env_names()
        assert "QDRANT_URL" not in names, "use VECTOR_DB_URL (the Settings field name)"
        assert "VECTOR_DB_URL" in names, (
            "the manifest must point pods at the vector store — without "
            "VECTOR_DB_URL the retrieval stack boots on its default address"
        )
