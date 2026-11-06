"""fly.toml matches the spec's deployment settings and the Dockerfile (no network)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fly() -> dict:
    with open(ROOT / "fly.toml", "rb") as f:
        return tomllib.load(f)


def test_scale_to_zero_settings():
    svc = _fly()["http_service"]
    assert svc["auto_stop_machines"] == "stop"
    assert svc["auto_start_machines"] is True
    assert svc["min_machines_running"] == 0
    assert svc["force_https"] is True


def test_machine_is_shared_cpu_512mb():
    (vm,) = _fly()["vm"]
    assert vm["memory"] == "512mb"
    assert vm["size"].startswith("shared-cpu")


def test_port_matches_dockerfile_and_env():
    fly = _fly()
    dockerfile = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^\s*PORT=(\d+)", dockerfile, re.M)
    assert match is not None
    docker_port = int(match.group(1))
    assert fly["http_service"]["internal_port"] == docker_port
    assert int(fly["env"]["PORT"]) == docker_port


def test_builds_the_repo_dockerfile():
    assert _fly()["build"]["dockerfile"] == "Dockerfile"
    assert (ROOT / "Dockerfile").is_file()


def test_health_check_hits_an_existing_route():
    from predict import app

    (check,) = _fly()["http_service"]["checks"]
    routes = {getattr(r, "path", None) for r in app.routes}
    assert check["path"] in routes
    assert check["method"] == "GET"


def test_make_deploy_runs_fly_deploy():
    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^deploy:.*\n\tfly deploy$", makefile, re.M)
