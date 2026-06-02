from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synto.config import Config
from synto.ollama_client import OllamaClient
from synto.state import StateDB


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Minimal vault structure for testing."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def make_mock_client(response: str = "{}") -> OllamaClient:
    """Return a mock OllamaClient that returns a fixed string from generate()."""
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = response
    client.embed_batch.return_value = [[0.1] * 768]
    client.embed.return_value = [0.1] * 768
    client.healthcheck.return_value = True
    return client


class RouterClient:
    """Test double that is both a mock LLM client and a ModelRouter.

    The per-role refactor changed pipeline entrypoints to take a ModelRouter instead of a
    client. Wrapping a mock client in this lets the existing call sites (and `.generate`
    assertions) stay unchanged: `endpoint(role).client` is the wrapped mock, and any other
    attribute (`generate`, `embed_batch`, side_effect, call_count, ...) delegates to it.
    """

    def __init__(self, client, config=None, model: str = "test-model", ctx: int = 8192):
        self._client = client
        self._config = config
        self._model = model
        self._ctx = ctx

    def endpoint(self, role):
        from synto.client_factory import RoleEndpoint

        if self._config is not None:
            rm = self._config.resolve_role(role)
            return RoleEndpoint(
                self._client, rm.model, rm.ctx, rm.think, rm.temperature, rm.options
            )
        return RoleEndpoint(self._client, self._model, self._ctx, None, None, {})

    def require_healthy(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_client"), name)


def as_router(client, config=None, model: str = "test-model", ctx: int = 8192) -> RouterClient:
    """Wrap a mock client so it can be passed where a ModelRouter is expected.

    Pass `config` so per-role ctx/model/think reflect the vault config (needed by tests
    that drive chunking via fast_ctx); otherwise a fixed default is used.
    """
    return RouterClient(client, config, model, ctx)


def as_endpoint(client, model: str = "test-model", ctx: int = 8192):
    """Wrap a mock client in a RoleEndpoint (for direct _query_core / extract_terms calls)."""
    from synto.client_factory import RoleEndpoint

    return RoleEndpoint(client, model, ctx, None, None, {})
