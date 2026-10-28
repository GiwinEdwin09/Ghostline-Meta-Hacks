"""Auth and the OpenAPI contract. None of these tests need a database."""

import dataclasses
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from ghostline.api.app import create_app
from ghostline.api.deps import FailureLimiter
from ghostline.config import get_settings


@pytest.fixture
def bare_client(settings):
    """The real app with no database: enough to test everything that happens before one."""
    return TestClient(create_app(settings=settings))


def schema(settings):
    return create_app(settings=settings).openapi()


# --- the endpoint list from Part 21 ---------------------------------------------------------
PART_21_ROUTES = {
    ("post", "/api/experiments"),
    ("post", "/api/uploads"),
    ("put", "/api/uploads/{upload_id}/chunks/{n}"),
    ("post", "/api/uploads/{upload_id}/complete"),
    ("post", "/api/trials/{trial_id}/trim"),
    ("post", "/api/experiments/{experiment_id}/analyze"),
    ("post", "/api/experiments/{experiment_id}/prove"),
    ("post", "/api/experiments/{experiment_id}/revise"),
    ("post", "/api/experiments/{experiment_id}/simulate"),
    ("get", "/api/experiments/{experiment_id}/export"),
    ("post", "/api/experiments/{experiment_id}/export/verify"),
    ("post", "/api/experiments/{experiment_id}/models/{model_id}/rerun"),
    ("get", "/api/experiments/{experiment_id}"),
    ("get", "/api/experiments/{experiment_id}/events"),
    ("get", "/api/experiments"),
    ("get", "/api/media/{experiment_id}/{path}"),
    ("get", "/api/benchmark"),
    ("put", "/api/benchmark/{experiment_id}"),
    ("get", "/api/health"),
}


def test_every_route_in_part_21_exists(settings):
    present = {(m, p) for p, ops in schema(settings)["paths"].items() for m in ops}

    assert PART_21_ROUTES <= present


def test_operation_ids_are_unique_and_readable(settings):
    ids = [op["operationId"] for ops in schema(settings)["paths"].values() for op in ops.values()]

    assert len(ids) == len(set(ids))
    assert all(i.isidentifier() and i == i.lower() for i in ids)


def test_the_openapi_document_carries_the_appendix_c_shapes(settings):
    components = schema(settings)["components"]["schemas"]

    for name in (
        "SceneAnalysis",
        "TrackedObject",
        "ParamSpec",
        "ModelProposal",
        "Quantity",
        "Conclusion",
        "ReviewIssue",
        "Review",
        "ExportedParam",
        "SimExport",
    ):
        assert name in components, name


def test_responses_are_typed_models(settings):
    """The TypeScript types come from the response schemas, so none may be bare JSON."""
    untyped = []
    for path, ops in schema(settings)["paths"].items():
        for method, op in ops.items():
            if path.startswith("/api/media"):
                continue
            ok = (
                op["responses"].get("200")
                or op["responses"].get("201")
                or op["responses"].get("202")
            )
            content = ok["content"]["application/json"]["schema"]
            if "$ref" not in content and content.get("type") != "array":
                untyped.append((method, path))

    assert untyped == []


def test_error_responses_are_declared(settings):
    op = schema(settings)["paths"]["/api/experiments/{experiment_id}/analyze"]["post"]

    assert op["responses"]["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorBody"
    }


def test_upload_and_experiment_request_fields_follow_the_plan(settings):
    components = schema(settings)["components"]["schemas"]

    assert set(components["UploadCreate"]["properties"]) == {
        "experiment_id",
        "role",
        "filename",
        "size",
    }
    assert {"title", "system_hint", "description", "declared_capture", "setup"} == set(
        components["ExperimentCreate"]["properties"]
    )
    assert {"plane_offset_cm", "paper", "pendulum_length_cm"} == set(
        components["Setup"]["properties"]
    )


# --- the passcode ---------------------------------------------------------------------------
def write_routes(settings):
    for path, ops in schema(settings)["paths"].items():
        for method in ops:
            if method not in ("get", "head"):
                yield method, path


def fill(path: str) -> str:
    for name in ("experiment_id", "upload_id", "trial_id", "job_id"):
        path = path.replace("{" + name + "}", str(uuid4()))
    return path.replace("{n}", "0").replace("{model_id}", "P1")


def test_every_write_route_refuses_requests_without_the_passcode(bare_client, settings):
    routes = list(write_routes(settings))
    assert len(routes) >= 12

    for method, path in routes:
        response = bare_client.request(method, fill(path), json={})
        assert response.status_code == 401, (method, path)
        assert response.json()["code"] == "passcode_required"


def test_every_write_route_refuses_a_wrong_passcode(bare_client, settings):
    for method, path in write_routes(settings):
        response = bare_client.request(
            method, fill(path), json={}, headers={"X-Ghostline-Key": "wrong"}
        )
        assert response.status_code == 401, (method, path)
        assert response.json()["code"] == "passcode_invalid"


def test_the_auth_check_route_reports_the_passcode(bare_client, auth):
    assert bare_client.get("/api/auth/check").status_code == 401
    assert bare_client.get("/api/auth/check", headers={"X-Ghostline-Key": "no"}).status_code == 401
    ok = bare_client.get("/api/auth/check", headers=auth)
    assert ok.status_code == 200 and ok.json() == {"ok": True}


def test_a_passcode_with_odd_characters_is_just_wrong(bare_client):
    response = bare_client.get("/api/auth/check", headers={"X-Ghostline-Key": "pässcode".encode()})

    assert response.status_code == 401


def test_no_configured_passcode_closes_write_routes(settings):
    closed = TestClient(create_app(settings=dataclasses.replace(settings, app_passcode="")))

    for key in ("", "anything"):
        response = closed.post("/api/experiments", json={}, headers={"X-Ghostline-Key": key})
        assert response.status_code == 503
        assert response.json()["code"] == "passcode_not_configured"


def test_guessing_the_passcode_gets_locked_out(settings, auth):
    app = create_app(settings=dataclasses.replace(settings, passcode_failures_per_10min=3))
    client = TestClient(app)

    codes = [
        client.get("/api/auth/check", headers={"X-Ghostline-Key": f"guess{i}"}).status_code
        for i in range(4)
    ]
    right_key_too = client.get("/api/auth/check", headers=auth)

    assert codes == [401, 401, 401, 429]
    assert right_key_too.status_code == 429  # locked out until the window passes


def test_lockout_is_per_client(settings, auth):
    client = TestClient(
        create_app(settings=dataclasses.replace(settings, passcode_failures_per_10min=2))
    )
    for _ in range(3):
        client.get(
            "/api/auth/check", headers={"X-Ghostline-Key": "x", "X-Forwarded-For": "1.1.1.1"}
        )

    other = client.get("/api/auth/check", headers={**auth, "X-Forwarded-For": "2.2.2.2"})

    assert other.status_code == 200


def test_failure_limiter_forgets_old_failures():
    now = [0.0]
    limiter = FailureLimiter(2, window_s=10, clock=lambda: now[0])
    limiter.record("a")
    limiter.record("a")
    assert limiter.blocked("a") and not limiter.blocked("b")

    now[0] = 11.0

    assert not limiter.blocked("a")


def test_the_passcode_is_never_in_a_response(bare_client, passcode):
    response = bare_client.get("/api/health")

    assert passcode not in response.text


def test_with_the_passcode_but_no_database_routes_say_so(bare_client, auth):
    response = bare_client.post("/api/experiments", json={"title": "x"}, headers=auth)

    assert response.status_code == 503
    assert response.json()["code"] == "database_not_configured"


# --- reads and limits that need no database ---------------------------------------------------
def test_json_bodies_over_the_limit_are_refused(settings, auth):
    client = TestClient(
        create_app(settings=dataclasses.replace(settings, json_body_limit_bytes=100))
    )

    response = client.post("/api/experiments", content=b"x" * 500, headers=auth)

    assert response.status_code == 413
    assert response.json()["code"] == "body_too_large"


def test_cors_allows_the_configured_origins_and_exposes_range_headers(settings):
    client = TestClient(
        create_app(settings=dataclasses.replace(settings, cors_origins=["https://app.example"]))
    )

    ok = client.options(
        "/api/health",
        headers={"Origin": "https://app.example", "Access-Control-Request-Method": "GET"},
    )
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    exposed = client.get("/api/health", headers={"Origin": "https://app.example"})

    assert ok.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in other.headers
    assert "Content-Range" in exposed.headers["access-control-expose-headers"]


# --- health ---------------------------------------------------------------------------------
def test_health_is_ok_with_nothing_configured(bare_client):
    body = bare_client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "unconfigured"
    assert body["meta_api"] == "unconfigured"
    assert {"version", "llm_mode", "volume"} <= set(body)


def test_health_reports_the_volume(settings, tmp_path):
    missing = dataclasses.replace(settings, volume_root=str(tmp_path / "nowhere"))

    assert (
        TestClient(create_app(settings=missing)).get("/api/health").json()["volume"]
        == "unconfigured"
    )
    assert TestClient(create_app(settings=settings)).get("/api/health").json()["volume"] == "ok"


def test_meta_api_reachability_is_cached(settings):
    calls = []
    results = iter([True, False, False])

    def ping():
        calls.append(1)
        return next(results)

    live = dataclasses.replace(settings, llm_mode="live", meta_health_ttl_s=60)
    client = TestClient(create_app(settings=live, meta_ping=ping))

    bodies = [client.get("/api/health").json() for _ in range(3)]

    assert len(calls) == 1
    assert [b["meta_api"] for b in bodies] == ["ok", "ok", "ok"]


def test_meta_api_down_degrades_health_but_replay_mode_skips_the_check(settings):
    live = dataclasses.replace(settings, llm_mode="live", meta_health_ttl_s=0)
    down = TestClient(create_app(settings=live, meta_ping=lambda: False)).get("/api/health").json()
    replay = dataclasses.replace(settings, llm_mode="replay")
    skipped = (
        TestClient(create_app(settings=replay, meta_ping=lambda: False)).get("/api/health").json()
    )

    assert (down["status"], down["meta_api"]) == ("degraded", "error")
    assert (skipped["status"], skipped["meta_api"]) == ("ok", "skipped")


def test_settings_have_sane_defaults():
    s = get_settings()

    assert s.upload_chunk_bytes == 8 * 1024 * 1024
    assert s.max_upload_bytes == 1024**3
    assert s.max_frames == 15_000
    assert s.analyses_per_ip_per_day == 10
    assert s.job_heartbeat_interval_s == 5 and s.job_heartbeat_timeout_s == 60
    assert s.volume_root == "/data"
