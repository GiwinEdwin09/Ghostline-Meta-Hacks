import base64
import hashlib
import json

from ghostline.llm import ReplayCache, normalize_request, request_hash


def test_secrets_are_removed_at_any_depth():
    request = {
        "model": "m",
        "api_key": "sk-secret",
        "headers": {"Authorization": "Bearer abc", "x-api-key": "k", "ok": 1},
        "input": [{"token": "t", "text": "hi"}],
    }

    normalized = normalize_request(request)

    assert "sk-secret" not in json.dumps(normalized)
    assert normalized == {"model": "m", "headers": {"ok": 1}, "input": [{"text": "hi"}]}


def test_images_become_content_hashes():
    payload = base64.b64encode(b"pixels").decode()
    request = {"image_url": f"data:image/png;base64,{payload}"}

    normalized = normalize_request(request)

    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert normalized == {"image_url": f"data:image/png;sha256={digest}"}


def test_file_ids_map_to_content_hashes_so_reuploads_share_a_key():
    first = normalize_request({"file_id": "file-aaa"}, {"file-aaa": "abc"})
    second = normalize_request({"file_id": "file-bbb"}, {"file-bbb": "abc"})

    assert first == second == {"file_id": "sha256:abc"}
    assert normalize_request({"file_id": "file-unknown"}) == {"file_id": "file-unknown"}


def test_hash_ignores_key_order_and_depends_on_content_and_kind():
    a = request_hash("respond", {"model": "m", "input": "x"})
    b = request_hash("respond", {"input": "x", "model": "m"})

    assert a == b
    assert a != request_hash("respond", {"model": "m", "input": "y"})
    assert a != request_hash("chat", {"model": "m", "input": "x"})


def test_cache_round_trip(tmp_path):
    cache = ReplayCache(tmp_path)

    cache.put("respond", "k1", {"model": "m", "request": {"a": 1}, "response": {"text": "hi"}})

    entry = cache.get("respond", "k1")
    assert entry["response"] == {"text": "hi"}
    assert entry["kind"] == "respond" and entry["hash"] == "k1"
    assert (tmp_path / "llm" / "k1.json").exists()
    assert cache.get("respond", "missing") is None


def test_sam_events_go_to_their_own_jsonl_file(tmp_path):
    cache = ReplayCache(tmp_path)
    events = [{"type": "response.created"}, {"type": "response.output_text.delta", "delta": "x"}]

    cache.put("sam_video", "k2", {"response": {"output_text": "x"}, "events": events})

    lines = (tmp_path / "sam" / "k2.events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == events
    assert "events" not in json.loads((tmp_path / "sam" / "k2.json").read_text())
    assert cache.get("sam_video", "k2")["events"] == events
