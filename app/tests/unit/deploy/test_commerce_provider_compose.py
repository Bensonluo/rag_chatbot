"""Compose contract for the internal commerce facts provider.

The provider must stay off the host network and share the same explicit token
source as the API. This guards against environment-name drift that would make
every external commerce call fail closed in deployment.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[4]
_COMPOSE_FILE = _REPO_ROOT / "docker-compose.yml"
_DEV_TOKEN_DEFAULT = "commerce-provider-dev-token"


def test_commerce_provider_is_internal_and_shares_api_token() -> None:
    config = yaml.safe_load(_COMPOSE_FILE.read_text())
    provider = config["services"]["commerce-provider"]
    api = config["services"]["api"]

    assert provider.get("ports", []) == []
    assert api["environment"]["COMMERCE_PROVIDER_MODE"] == "external"
    assert api["environment"]["COMMERCE_PROVIDER_URL"] == "http://commerce-provider:8000"
    assert (
        provider["environment"]["COMMERCE_PROVIDER_AUTH_TOKEN"]
        == f"${{COMMERCE_PROVIDER_TOKEN:-{_DEV_TOKEN_DEFAULT}}}"
    )
    assert (
        api["environment"]["COMMERCE_PROVIDER_TOKEN"]
        == f"${{COMMERCE_PROVIDER_TOKEN:-{_DEV_TOKEN_DEFAULT}}}"
    )


def test_commerce_provider_state_is_durable_and_faults_disabled() -> None:
    config = yaml.safe_load(_COMPOSE_FILE.read_text())
    provider = config["services"]["commerce-provider"]

    assert provider["environment"]["COMMERCE_PROVIDER_DATABASE_PATH"] == "/data/commerce.db"
    assert provider["environment"]["COMMERCE_PROVIDER_FAULTS_ENABLED"] == "false"
    assert "commerce_provider_data:/data" in provider["volumes"]
    assert "commerce_provider_data" in config["volumes"]
