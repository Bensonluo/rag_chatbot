"""Exercise deployment orchestration with command doubles; never operate Docker or git."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path("deployment/production/scripts")

COMMAND_DOUBLE = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

command = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'command': command, 'args': args,
        'compose_file': os.environ.get('COMPOSE_FILE'),
        'db_service': os.environ.get('DB_SERVICE')}) + '\n')
if command == 'backup.sh':
    if os.environ.get('FAIL_BACKUP') == '1':
        sys.exit(4)
    if os.environ.get('BACKUP_DIR'):
        Path(os.environ['BACKUP_DIR']).mkdir(parents=True, exist_ok=True)
        (Path(os.environ['BACKUP_DIR']) / 'database.sql.gz').write_text('fixture backup')
    sys.exit(0)
if command == 'sleep':
    sys.exit(0)
if command == 'curl':
    sys.exit(22 if os.environ.get('FAIL_READY') == '1' else 0)
if command == 'git':
    if 'rev-parse' in args:
        print('prior-revision')
    if 'pull' in args and os.environ.get('FAIL_PULL') == '1':
        sys.exit(5)
    sys.exit(0)
if command in ('docker', 'docker-compose'):
    if 'version' in args:
        sys.exit(1 if command == 'docker' and os.environ.get('FAIL_PLUGIN') == '1' else 0)
    if 'inspect' in args:
        if '--format' in args and '.Aliases' in args[args.index('--format') + 1]:
            print('/rag-chatbot-db\n' + os.environ.get('DB_SERVICE', 'db'))
        elif '--format' in args and '.IPAddress' in args[args.index('--format') + 1]:
            print('172.18.0.2')
        elif '--format' in args:
            stopped = os.environ.get('DB_STATE') == 'stopped' and args[-1] == 'prior-db'
            stopped = stopped or (os.environ.get('APP_STATE') == 'stopped' and args[-1] == 'prior-app')
            print('false' if stopped else 'true')
        else:
            print('[{"Id":"prior-container","Image":"sha256:prior-image"}]')
        sys.exit(0)
    if 'config' in args:
        if '--quiet' not in args and '-q' not in args:
            print('services: {} # protected configuration secret-token')
        sys.exit(0)
    if 'ps' in args:
        if os.environ.get('DB_STATE') != 'absent':
            if args[-1] in ('app', 'api'):
                if os.environ.get('APP_STATE') != 'absent':
                    print('prior-app')
            elif args[-1] in ('db', 'postgres'):
                print('prior-db' if '-q' in args else 'database Up')
            else:
                print('prior-db\nprior-app')
        sys.exit(0)
    if 'images' in args:
        print('sha256:prior-image')
        sys.exit(0)
    if 'pg_dump' in ' '.join(args):
        if os.environ.get('FAIL_BACKUP') == '1':
            sys.exit(4)
        print('-- database dump')
    if 'pg_isready' in ' '.join(args) and os.environ.get('FAIL_DB') == '1':
        sys.exit(1)
    if 'curl' in args:
        sys.exit(subprocess.call(['curl', *args[args.index('curl') + 1:]]))
    if 'SHOW port' in ' '.join(args):
        print('rag_chatbot\n5432')
        sys.exit(0)
    if 'python' in args and 'BACKUP_TARGET_CHECK' in ' '.join(args):
        import socket
        import types
        settings = types.SimpleNamespace(
            DATABASE_URL=os.environ.get('TEST_DATABASE_URL', 'postgresql://app:secret-token@' + os.environ['DB_SERVICE'] + ':5432/rag_chatbot'),
            DIALOGUE_CHECKPOINTER_TYPE=os.environ.get('TEST_CHECKPOINTER_TYPE', 'postgres'),
            DIALOGUE_CHECKPOINTER_DB_URL=os.environ.get('TEST_CHECKPOINTER_URL'),
        )
        def get_settings():
            if os.environ.get('TEST_SETTINGS_ERROR'):
                raise ValueError('secret-token')
            return settings
        for name in ('app', 'app.config', 'app.config.settings'):
            sys.modules[name] = types.ModuleType(name)
        sys.modules['app.config.settings'].get_settings = get_settings
        socket.getaddrinfo = lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (os.environ.get('TEST_DNS_ADDRESS', '172.18.0.2'), 5432))]
        code_index = args.index('-c')
        sys.argv = ['check-target', *args[code_index + 2:]]
        exec(args[code_index + 1])
        sys.exit(0)
    if 'python' in args and os.environ.get('FAIL_REDIS') == '1':
        sys.exit(1)
    if 'build' in args and os.environ.get('FAIL_BUILD') == '1':
        sys.exit(6)
    if 'up' in args and os.environ.get('FAIL_UP') == '1':
        sys.exit(7)
    sys.exit(0)
sys.exit('Unexpected command double')
"""


class CommandCall(TypedDict):
    command: str
    args: list[str]
    compose_file: str | None
    db_service: str | None


@dataclass
class DeploymentHarness:
    root: Path
    log: Path
    env: dict[str, str]

    def run(self, script: str, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / SCRIPTS / script), *args],
            cwd=self.root.parent,
            env={**self.env, **overrides},
            input="y\n",
            text=True,
            capture_output=True,
            timeout=30,
        )

    def calls(self) -> list[CommandCall]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def deployment(tmp_path: Path) -> DeploymentHarness:
    project = tmp_path / "project with spaces"
    scripts = project / SCRIPTS
    scripts.mkdir(parents=True)
    for name in ("deploy.sh", "update.sh"):
        shutil.copy2(ROOT / SCRIPTS / name, scripts / name)
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        shutil.copy2(ROOT / name, project / name)
    (project / ".env").write_text("REDIS_PASSWORD=secret-token\n")
    binaries = tmp_path / "commands"
    binaries.mkdir()
    for name in ("docker", "docker-compose", "curl", "git", "sleep"):
        path = binaries / name
        path.write_text(COMMAND_DOUBLE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        path.chmod(0o755)
    backup = scripts / "backup.sh"
    backup.write_text(COMMAND_DOUBLE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    backup.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    log.touch()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COMPOSE_", "DB_", "APP_", "DEPLOY_", "READINESS_", "FAIL_"))
        and key not in ("ENV_FILE", "BACKUP_DIR", "PROJECT_DIRECTORY", "ALLOW_INITIAL_DEPLOY")
    }
    env.update(
        PATH=f"{binaries}{os.pathsep}{env['PATH']}",
        CALL_LOG=str(log),
        READINESS_ATTEMPTS="2",
        READINESS_INTERVAL="0",
    )
    return DeploymentHarness(project, log, env)


def test_prod_deploy_selects_database_before_backup(deployment: DeploymentHarness) -> None:
    result = deployment.run("deploy.sh", "prod")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = deployment.calls()
    backup = next(call for call in calls if call["command"] == "backup.sh")
    assert backup["compose_file"] == str(deployment.root / "docker-compose.prod.yml")
    assert backup["db_service"] == "db"
    assert all("docker-compose.yml" not in str(call["args"]) for call in calls)
    assert not any("pull" in call["args"] for call in calls)


def test_update_passes_production_backup_context(deployment: DeploymentHarness) -> None:
    result = deployment.run("update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    backup = next(call for call in deployment.calls() if call["command"] == "backup.sh")
    assert backup["compose_file"] == str(deployment.root / "docker-compose.prod.yml")
    assert backup["db_service"] == "db"


def test_update_readiness_timeout_is_failure(deployment: DeploymentHarness) -> None:
    result = deployment.run("update.sh", FAIL_READY="1")
    assert result.returncode != 0
    assert "Successfully" not in result.stdout


def test_failed_update_does_not_restart_dependencies_or_claim_rollback(
    deployment: DeploymentHarness,
) -> None:
    result = deployment.run("update.sh", FAIL_UP="1")
    assert result.returncode != 0
    calls = deployment.calls()
    assert not any("down" in call["args"] for call in calls)
    assert "Rollback completed" not in result.stdout


@pytest.mark.parametrize("script,args", [("deploy.sh", ("prod",)), ("update.sh", ())])
@pytest.mark.parametrize("failure", ["FAIL_BACKUP", "FAIL_BUILD", "FAIL_DB", "FAIL_REDIS"])
def test_failure_retains_state_without_teardown(
    deployment: DeploymentHarness, script: str, args: tuple[str, ...], failure: str
) -> None:
    result = deployment.run(script, *args, **{failure: "1"})
    assert result.returncode != 0
    calls = deployment.calls()
    assert not any("down" in call["args"] for call in calls)
    snapshots = list((deployment.root / "backups").glob("deployment_*"))
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert "failed" in (snapshot / "result.txt").read_text()
    assert (snapshot / "environment.env").read_text() == "REDIS_PASSWORD=secret-token\n"
    assert (snapshot / "image-ids.txt").read_text().strip() == "sha256:prior-image"
    assert (snapshot / "git-revision.txt").read_text().strip() == "prior-revision"
    assert "db_service=db" in (snapshot / "deployment-context.txt").read_text()
    assert snapshot.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in snapshot.iterdir())
    assert "secret-token" not in result.stdout + result.stderr
    if failure == "FAIL_BACKUP":
        assert not any("pull" in call["args"] or "build" in call["args"] for call in calls)


def test_probes_target_selected_services_without_host_credentials(
    deployment: DeploymentHarness,
) -> None:
    result = deployment.run("deploy.sh", "prod", REDIS_PASSWORD="secret-token")
    assert result.returncode == 0, result.stderr
    calls = deployment.calls()
    probes = [call for call in calls if "exec" in call["args"]]
    postgres = next(call for call in probes if "pg_isready" in " ".join(call["args"]))
    assert postgres["args"][postgres["args"].index("exec") + 2] == "db"
    redis = next(call for call in probes if "REDIS_URL" in " ".join(call["args"]))
    assert "REDIS_URL" in " ".join(redis["args"])
    assert "secret-token" not in " ".join(redis["args"])
    ready = next(call for call in probes if "curl" in call["args"])
    assert ready["args"][ready["args"].index("exec") + 2] == "app"
    assert "http://localhost:8000/ready" in ready["args"]
    assert "--max-time" in ready["args"]
    assert not any("/health" in " ".join(call["args"]) for call in calls)


def test_dev_services_and_compose_v1_remain_supported(deployment: DeploymentHarness) -> None:
    result = deployment.run("deploy.sh", "dev", FAIL_PLUGIN="1")
    assert result.returncode == 0, result.stderr
    calls = deployment.calls()
    assert any(call["command"] == "docker-compose" for call in calls)
    backup = next(call for call in calls if call["command"] == "backup.sh")
    assert backup["db_service"] == "postgres"
    assert backup["compose_file"] == str(deployment.root / "docker-compose.yml")
    ready = next(call for call in calls if "curl" in call["args"])
    assert ready["args"][ready["args"].index("exec") + 2] == "api"


@pytest.mark.parametrize("state", ["absent", "stopped"])
def test_prod_requires_backup_before_mutating_services(
    deployment: DeploymentHarness, state: str
) -> None:
    result = deployment.run("deploy.sh", "prod", DB_STATE=state)
    assert result.returncode != 0
    assert not any("up" in call["args"] or "build" in call["args"] for call in deployment.calls())


def test_initial_prod_requires_explicit_backup_skip(deployment: DeploymentHarness) -> None:
    result = deployment.run("deploy.sh", "prod", DB_STATE="absent", ALLOW_INITIAL_DEPLOY="1")
    assert result.returncode == 0, result.stderr
    assert "Existing volumes have not been backed up" in result.stdout
    assert not any(call["command"] == "backup.sh" for call in deployment.calls())


def test_update_cannot_skip_missing_database_backup(deployment: DeploymentHarness) -> None:
    result = deployment.run("update.sh", DB_STATE="absent", ALLOW_INITIAL_DEPLOY="1")
    assert result.returncode != 0
    assert not any("pull" in call["args"] or "up" in call["args"] for call in deployment.calls())


def test_git_failure_preserves_state_and_never_replaces_application(
    deployment: DeploymentHarness,
) -> None:
    result = deployment.run("update.sh", FAIL_PULL="1")
    assert result.returncode != 0
    calls = deployment.calls()
    pull = next(call for call in calls if call["command"] == "git" and "pull" in call["args"])
    assert "--ff-only" in pull["args"]
    assert not any("up" in call["args"] for call in calls)


def test_real_backup_script_receives_production_context(deployment: DeploymentHarness) -> None:
    shutil.copy2(ROOT / SCRIPTS / "backup.sh", deployment.root / SCRIPTS / "backup.sh")
    result = deployment.run("update.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    dumps = [call for call in deployment.calls() if "pg_dump" in " ".join(call["args"])]
    assert len(dumps) == 1
    assert dumps[0]["args"][dumps[0]["args"].index("exec") + 2] == "db"
    assert str(deployment.root / "docker-compose.prod.yml") in dumps[0]["args"]
    archives = list((deployment.root / "backups").glob("deployment_*/database/*.sql.gz"))
    assert len(archives) == 1


def test_external_compose_and_backup_share_project_identity(deployment: DeploymentHarness) -> None:
    external = deployment.root.parent / "external configuration"
    external.mkdir()
    compose_file = external / "production.yml"
    shutil.copy2(deployment.root / "docker-compose.prod.yml", compose_file)
    shutil.copy2(ROOT / SCRIPTS / "backup.sh", deployment.root / SCRIPTS / "backup.sh")
    result = deployment.run("update.sh", COMPOSE_FILE=str(compose_file))
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [
        call for call in deployment.calls() if call["command"] == "docker" and "-f" in call["args"]
    ]
    assert calls
    for call in calls:
        args = call["args"]
        assert args[args.index("-f") + 1] == str(compose_file)
        assert "--project-directory" in args
        assert args[args.index("--project-directory") + 1] == str(deployment.root)
    assert any("pg_dump" in " ".join(call["args"]) for call in calls)


def test_relative_selectors_resolve_before_changing_directory(
    deployment: DeploymentHarness,
) -> None:
    external = deployment.root.parent / "relative-config"
    external.mkdir()
    compose_file = external / "production.yml"
    env_file = external / "selected.env"
    shutil.copy2(deployment.root / "docker-compose.prod.yml", compose_file)
    env_file.write_text("REDIS_PASSWORD=secret-token\n")
    result = deployment.run(
        "update.sh",
        COMPOSE_FILE="relative-config/production.yml",
        ENV_FILE="relative-config/selected.env",
        PROJECT_DIRECTORY="relative-config",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [
        call for call in deployment.calls() if call["command"] == "docker" and "-f" in call["args"]
    ]
    assert calls
    for call in calls:
        args = call["args"]
        assert args[args.index("-f") + 1] == str(compose_file)
        assert args[args.index("--env-file") + 1] == str(env_file)
        assert args[args.index("--project-directory") + 1] == str(external)
    backup = next(call for call in deployment.calls() if call["command"] == "backup.sh")
    assert backup["compose_file"] == str(compose_file)


@pytest.mark.parametrize(
    "override",
    [
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@external:5432/rag_chatbot"},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@db:5432/other"},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@db:5433/rag_chatbot"},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@localhost:5432/rag_chatbot"},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@db:5432/rag_chatbot?host=external"},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@db:5432/rag_chatbot?host="},
        {"TEST_DATABASE_URL": "postgresql://user:secret-token@db:0/rag_chatbot"},
        {"TEST_DATABASE_URL": "malformed://secret-token"},
        {"TEST_CHECKPOINTER_URL": "postgresql://user:secret-token@db:5432/checkpoints"},
        {"TEST_CHECKPOINTER_URL": "postgresql://user:secret-token@external:5432/rag_chatbot"},
        {"TEST_DNS_ADDRESS": "192.0.2.8"},
        {"TEST_SETTINGS_ERROR": "1"},
        {"APP_STATE": "stopped"},
    ],
)
def test_unverified_application_database_cannot_be_backed_up_as_local(
    deployment: DeploymentHarness,
    override: dict[str, str],
) -> None:
    result = deployment.run("update.sh", **override)
    assert result.returncode != 0
    assert "secret-token" not in result.stdout + result.stderr
    calls = deployment.calls()
    assert not any(call["command"] == "backup.sh" for call in calls)
    assert not any(
        "pull" in call["args"] or "build" in call["args"] or "up" in call["args"] for call in calls
    )


@pytest.mark.parametrize(
    "override",
    [
        {"TEST_DATABASE_URL": "postgresql+asyncpg://different:secret-token@db:5432/rag_chatbot"},
        {
            "TEST_DATABASE_URL": "postgresql://different:secret-token@rag-chatbot-db:5432/rag_chatbot"
        },
        {"TEST_CHECKPOINTER_TYPE": "memory", "TEST_CHECKPOINTER_URL": "postgresql://other/db"},
    ],
)
def test_same_target_aliases_users_and_memory_checkpoints_are_supported(
    deployment: DeploymentHarness,
    override: dict[str, str],
) -> None:
    result = deployment.run("update.sh", **override)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "secret-token" not in result.stdout + result.stderr
    snapshot = next((deployment.root / "backups").glob("deployment_*"))
    assert (
        snapshot / "backup-target.txt"
    ).read_text().strip() == "matched-running-application-database-and-checkpoint"


def test_absent_application_limits_backup_claim_to_local_database(
    deployment: DeploymentHarness,
) -> None:
    result = deployment.run("update.sh", APP_STATE="absent")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "only the selected local database backup can be verified" in result.stdout
    snapshot = next((deployment.root / "backups").glob("deployment_*"))
    assert "local-database-only" in (snapshot / "backup-target.txt").read_text()
