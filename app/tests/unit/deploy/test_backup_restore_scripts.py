"""Run disaster-recovery scripts against executable substitutes, never Docker."""

import gzip
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class ScriptFixture:
    root: Path
    env: dict[str, str]

    def run(self, name: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / "deployment/production/scripts" / name), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def calls(self) -> list[list[str]]:
        log = self.root / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def archive(self, content: bytes = b"CREATE TABLE recovered(id integer);\n") -> Path:
        path = self.root / "trusted.sql.gz"
        path.write_bytes(gzip.compress(content))
        return path


@pytest.fixture
def scripts(tmp_path: Path) -> ScriptFixture:
    root = tmp_path / "project"
    target = root / "deployment/production/scripts"
    target.mkdir(parents=True)
    for name in ("backup.sh", "restore.sh"):
        source = REPO_ROOT / "deployment/production/scripts" / name
        if source.exists():
            shutil.copy2(source, target / name)
    bins = root / "bin"
    bins.mkdir()
    executable = (
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps([name, *args]) + '\n')
if name in ('docker', 'docker-compose'):
    if 'version' in args:
        sys.exit(0)
    index = args.index('exec')
    assert args[index + 1] == '-T'
    command = args[index + 3:]
    env = dict(os.environ, POSTGRES_USER='container_owner', POSTGRES_DB='source_db')
    sys.exit(subprocess.run(command, env=env).returncode)
if name == 'pg_dump':
    print('-- fixture SQL')
    if os.environ.get('DUMP_EXIT', '0') != '0':
        print('fixture diagnostic: synthetic-private-token', file=sys.stderr)
    sys.exit(int(os.environ.get('DUMP_EXIT', '0')))
if name == 'createdb':
    marker = root / ('database-' + args[-1])
    if marker.exists():
        sys.exit(1)
    marker.touch()
    sys.exit(0)
if name == 'psql':
    (root / 'restored.sql').write_bytes(sys.stdin.buffer.read())
    sys.exit(int(os.environ.get('RESTORE_EXIT', '0')))
raise AssertionError(name)
"""
    )
    for name in ("docker", "docker-compose", "pg_dump", "createdb", "psql"):
        path = bins / name
        path.write_text(executable)
        path.chmod(0o755)
    # No real configuration, credentials, Docker executable, or database is used.
    env = {
        "PATH": f"{bins}:/usr/bin:/bin",
        "FIXTURE_ROOT": str(root),
        "POSTGRES_USER": "wrong_host_user",
        "POSTGRES_DB": "wrong_host_database",
        "COMPOSE_FILE": str(root / "chosen-compose.yml"),
        "DB_SERVICE": "db",
        "ENV_FILE": str(root / "fixture.env"),
        "BACKUP_DIR": str(root / "archives"),
    }
    (root / "fixture.env").write_text("# no credentials\n")
    (root / "scratch").mkdir()
    env["TMPDIR"] = str(root / "scratch")
    return ScriptFixture(root, env)


def test_backup_targets_explicit_compose_and_container_database(scripts: ScriptFixture) -> None:
    result = scripts.run("backup.sh")
    assert result.returncode == 0, result.stderr
    archives = list((scripts.root / "archives").glob("*.sql.gz"))
    assert len(archives) == 1
    assert gzip.decompress(archives[0].read_bytes()) == b"-- fixture SQL\n"
    assert archives[0].stat().st_mode & 0o777 == 0o600
    assert (scripts.root / "archives").stat().st_mode & 0o777 == 0o700
    calls = scripts.calls()
    dump = next(call for call in calls if call[0] == "pg_dump")
    assert "container_owner" in dump and "source_db" in dump
    assert "wrong_host_user" not in dump and "wrong_host_database" not in dump
    compose = next(call for call in calls if "exec" in call)
    assert compose[compose.index("-f") + 1] == scripts.env["COMPOSE_FILE"]
    assert compose[compose.index("--env-file") + 1] == scripts.env["ENV_FILE"]
    assert compose[compose.index("exec") + 2] == "db"
    assert not list((scripts.root / "archives").glob(".backup.*"))
    assert not list((scripts.root / "archives").glob("*.error.log"))


@pytest.mark.parametrize("retention", ["-1", "one", "1 -o -delete", ""])
def test_invalid_retention_never_touches_existing_backups(
    scripts: ScriptFixture, retention: str
) -> None:
    archives = scripts.root / "archives"
    archives.mkdir()
    old = archives / "db_backup_old.sql.gz"
    old.write_bytes(b"keep me")
    result = scripts.run("backup.sh", retention)
    assert result.returncode != 0
    assert old.read_bytes() == b"keep me"
    assert not any(call[0] == "pg_dump" for call in scripts.calls())


def test_failed_dump_keeps_old_backup_and_removes_partial(scripts: ScriptFixture) -> None:
    archives = scripts.root / "archives"
    archives.mkdir()
    old = archives / "db_backup_old.sql.gz"
    old.write_bytes(b"old good backup")
    os.utime(old, (0, 0))
    scripts.env["DUMP_EXIT"] = "1"
    result = scripts.run("backup.sh", "0")
    assert result.returncode != 0
    assert "Database backup failed" in result.stdout
    assert list(archives.glob("*.sql.gz")) == [old]
    errors = list(archives.glob("*.error.log"))
    assert len(errors) == 1
    assert errors[0].stat().st_mode & 0o777 == 0o600
    assert "fixture diagnostic: synthetic-private-token" in errors[0].read_text()
    assert str(errors[0]) in result.stdout
    assert "synthetic-private-token" not in result.stdout + result.stderr
    assert not list(archives.glob(".backup.*"))
    assert "Backup completed successfully" not in result.stdout


def test_backups_with_same_timestamp_do_not_overwrite(scripts: ScriptFixture) -> None:
    date = scripts.root / "bin/date"
    date.write_text("#!/bin/sh\nprintf '20261018_120000\\n'\n")
    date.chmod(0o755)
    assert scripts.run("backup.sh").returncode == 0
    assert scripts.run("backup.sh").returncode == 0
    assert len(list((scripts.root / "archives").glob("*.sql.gz"))) == 2


def test_backup_retention_removes_only_expired_archives(scripts: ScriptFixture) -> None:
    archives = scripts.root / "archives"
    archives.mkdir()
    old = archives / "db_backup_old.sql.gz"
    old.write_bytes(b"expired")
    os.utime(old, (0, 0))
    other = archives / "important.sql"
    other.write_text("keep")
    assert scripts.run("backup.sh", "1").returncode == 0
    assert not old.exists()
    assert other.read_text() == "keep"
    assert len(list(archives.glob("*.sql.gz"))) == 1


def test_restore_requires_explicit_new_target(scripts: ScriptFixture) -> None:
    result = scripts.run("restore.sh", str(scripts.archive()))
    assert result.returncode != 0
    assert not scripts.calls()


@pytest.mark.parametrize("database", ["source_db", "bad=name", "a" * 64, "-option"])
def test_restore_refuses_unsafe_or_source_target(scripts: ScriptFixture, database: str) -> None:
    result = scripts.run("restore.sh", str(scripts.archive()), "--database", database)
    assert result.returncode != 0
    assert not any(call[0] in {"createdb", "psql"} for call in scripts.calls())


def test_restore_rejects_corrupt_gzip_before_database_creation(scripts: ScriptFixture) -> None:
    archive = scripts.archive()
    archive.write_bytes(archive.read_bytes()[:-4])
    result = scripts.run("restore.sh", str(archive), "--database", "recovery_db")
    assert result.returncode != 0
    assert not any(call[0] in {"createdb", "psql"} for call in scripts.calls())


def test_restore_existing_database_cannot_be_overwritten(scripts: ScriptFixture) -> None:
    marker = scripts.root / "database-recovery_db"
    marker.write_text("existing rows")
    result = scripts.run("restore.sh", str(scripts.archive()), "--database", "recovery_db")
    assert result.returncode != 0
    assert marker.read_text() == "existing rows"
    assert not any(call[0] == "psql" for call in scripts.calls())


def test_restore_uses_one_transaction_and_container_identity(scripts: ScriptFixture) -> None:
    data = b"CREATE TABLE recovered(id integer);\nINSERT INTO recovered VALUES (7);\n"
    result = scripts.run("restore.sh", str(scripts.archive(data)), "--database", "recovery_db")
    assert result.returncode == 0, result.stderr
    calls = scripts.calls()
    create = next(call for call in calls if call[0] == "createdb")
    restore = next(call for call in calls if call[0] == "psql")
    assert "--template=template0" in create
    assert "container_owner" in create and "container_owner" in restore
    assert "--single-transaction" in restore and "-X" in restore
    assert "--set=ON_ERROR_STOP=1" in restore
    assert "--file=-" in restore
    assert "--dbname=recovery_db" in restore
    assert (scripts.root / "restored.sql").read_bytes() == data


def test_failed_restore_reports_failure_without_deleting_new_database(
    scripts: ScriptFixture,
) -> None:
    scripts.env["RESTORE_EXIT"] = "3"
    result = scripts.run("restore.sh", str(scripts.archive()), "--database", "recovery_db")
    assert result.returncode != 0
    assert (scripts.root / "database-recovery_db").exists()
    assert "Restore completed successfully" not in result.stdout


@pytest.mark.parametrize("empty", [False, True])
def test_empty_dump_or_compression_failure_never_publishes_archive(
    scripts: ScriptFixture, empty: bool
) -> None:
    binary = scripts.root / "bin" / ("pg_dump" if empty else "gzip")
    binary.write_text("#!/bin/sh\nexit 0\n" if empty else "#!/bin/sh\necho partial\nexit 1\n")
    binary.chmod(0o755)
    archives = scripts.root / "archives"
    archives.mkdir()
    old = archives / "db_backup_old.sql.gz"
    old.write_bytes(b"old good backup")
    os.utime(old, (0, 0))
    result = scripts.run("backup.sh", "0")
    assert result.returncode != 0
    assert list(archives.iterdir()) == [old]


def test_restore_decode_failure_after_integrity_check_cannot_create_database(
    scripts: ScriptFixture,
) -> None:
    # Simulate a file changing after gzip -t: complete SQL can be emitted before
    # a decoder fails. Never stream this prefix to a transaction that sees EOF.
    binary = scripts.root / "bin/gzip"
    binary.write_text(
        '#!/bin/sh\nif [ "$1" = -t ]; then exit 0; fi\n'
        "echo 'CREATE TABLE partial(id integer);'\nexit 1\n"
    )
    binary.chmod(0o755)
    result = scripts.run("restore.sh", str(scripts.archive()), "--database", "recovery_db")
    assert result.returncode != 0
    assert not any(call[0] in {"createdb", "psql"} for call in scripts.calls())
    assert list((scripts.root / "scratch").iterdir()) == []


def test_restore_explicit_environment_target_and_temporary_cleanup(scripts: ScriptFixture) -> None:
    scripts.env["RESTORE_DB"] = "environment_recovery"
    assert scripts.run("restore.sh", str(scripts.archive())).returncode == 0
    assert (scripts.root / "database-environment_recovery").exists()
    assert list((scripts.root / "scratch").iterdir()) == []


def test_retention_preserves_nested_deployment_and_environment_archives(
    scripts: ScriptFixture,
) -> None:
    archives = scripts.root / "archives"
    snapshots = [
        archives / "deployment_20261001/database/db_backup_snapshot.sql.gz",
        archives / "other-environment/db_backup_other.sql.gz",
    ]
    for snapshot in snapshots:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"nested recovery snapshot")
        os.utime(snapshot, (0, 0))
    direct = archives / "db_backup_expired.sql.gz"
    direct.write_bytes(b"expired standalone backup")
    os.utime(direct, (0, 0))

    result = scripts.run("backup.sh", "1")

    assert result.returncode == 0, result.stderr
    assert not direct.exists()
    for snapshot in snapshots:
        assert snapshot.read_bytes() == b"nested recovery snapshot"
    assert len(list(archives.glob("*.sql.gz"))) == 1


@pytest.mark.parametrize("name", ["backup.sh", "restore.sh"])
def test_external_compose_uses_explicit_project_directory(
    scripts: ScriptFixture, name: str
) -> None:
    scripts.env["PROJECT_DIRECTORY"] = str(scripts.root / "selected-project")
    scripts.env["COMPOSE_FILE"] = str(scripts.root / "external/production.yml")
    args = (str(scripts.archive()), "--database", "recovery_db") if name == "restore.sh" else ()

    result = scripts.run(name, *args)

    assert result.returncode == 0, result.stderr
    calls = [call for call in scripts.calls() if "exec" in call]
    assert calls
    for call in calls:
        assert call[call.index("--project-directory") + 1] == scripts.env["PROJECT_DIRECTORY"]
        assert call[call.index("-f") + 1] == scripts.env["COMPOSE_FILE"]
        assert "--project-name" not in call


@pytest.mark.parametrize("name", ["backup.sh", "restore.sh"])
def test_standalone_compose_keeps_file_relative_project_default(
    scripts: ScriptFixture, name: str
) -> None:
    args = (str(scripts.archive()), "--database", "recovery_db") if name == "restore.sh" else ()

    result = scripts.run(name, *args)

    assert result.returncode == 0, result.stderr
    calls = [call for call in scripts.calls() if "exec" in call]
    assert calls
    assert all("--project-directory" not in call for call in calls)
