import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from ghostline.api.app import create_app

VIDEO = os.urandom(300_000)


@pytest.fixture
def exp(repo, paths):
    experiment = repo.create_experiment(title="clip")
    folder = paths.experiment_dir(experiment.id)
    (folder / "trials" / "t1").mkdir(parents=True)
    (folder / "ghost").mkdir()
    (folder / "trials" / "t1" / "playback_720.mp4").write_bytes(VIDEO)
    (folder / "ghost" / "v1.mp4").write_bytes(VIDEO[:1000])
    (folder / "trials" / "t1" / "tracks.jsonl").write_bytes(b'{"a": 1}\n')
    (folder / "trials" / "t1" / "x.sam.events.jsonl").write_bytes(b'{"a": 1}\n')
    return experiment


def url(exp, path="trials/t1/playback_720.mp4"):
    return f"/api/media/{exp.id}/{path}"


# --- plain reads ----------------------------------------------------------------------------
def test_media_is_served_without_a_passcode(exp, anon):
    response = anon.get(url(exp))

    assert response.status_code == 200
    assert response.content == VIDEO
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == str(len(VIDEO))


def test_content_types(exp, anon):
    assert (
        anon.get(url(exp, "trials/t1/tracks.jsonl"))
        .headers["content-type"]
        .startswith("application/x-ndjson")
    )
    assert anon.get(url(exp, "trials/t1/x.sam.events.jsonl")).status_code == 200


# --- Range (Safari will not play video without it) ------------------------------------------
@pytest.mark.parametrize(
    "header, start, end",
    [
        ("bytes=0-1", 0, 1),  # Safari's first probe
        ("bytes=0-", 0, len(VIDEO) - 1),
        ("bytes=100000-199999", 100_000, 199_999),
        ("bytes=250000-", 250_000, len(VIDEO) - 1),
        ("bytes=-500", len(VIDEO) - 500, len(VIDEO) - 1),
        ("bytes=299999-299999", len(VIDEO) - 1, len(VIDEO) - 1),
        ("bytes=290000-9999999", 290_000, len(VIDEO) - 1),  # end past the file is clamped
    ],
)
def test_range_requests_get_exactly_the_bytes_asked_for(exp, anon, header, start, end):
    response = anon.get(url(exp), headers={"Range": header})

    assert response.status_code == 206
    assert response.content == VIDEO[start : end + 1]
    assert response.headers["content-range"] == f"bytes {start}-{end}/{len(VIDEO)}"
    assert response.headers["content-length"] == str(end - start + 1)
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"


def test_a_range_past_the_end_is_not_satisfiable(exp, anon):
    response = anon.get(url(exp), headers={"Range": f"bytes={len(VIDEO)}-"})

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(VIDEO)}"


def test_a_garbled_range_is_a_client_error(exp, anon):
    assert anon.get(url(exp), headers={"Range": "bytes=abc"}).status_code in (400, 416)


def test_several_ranges_come_back_as_multipart(exp, anon):
    response = anon.get(url(exp), headers={"Range": "bytes=0-9,20-29"})

    assert response.status_code == 206
    assert response.headers["content-type"].startswith("multipart/byteranges")
    assert VIDEO[:10] in response.content and VIDEO[20:30] in response.content


def test_if_range_with_a_matching_etag_returns_the_range_otherwise_the_whole_file(exp, anon):
    etag = anon.get(url(exp)).headers["etag"]

    match = anon.get(url(exp), headers={"Range": "bytes=0-9", "If-Range": etag})
    stale = anon.get(url(exp), headers={"Range": "bytes=0-9", "If-Range": '"old"'})

    assert match.status_code == 206 and len(match.content) == 10
    assert stale.status_code == 200 and stale.content == VIDEO


def test_head_reports_size_and_range_support_without_a_body(exp, anon):
    head = anon.head(url(exp))
    ranged = anon.head(url(exp), headers={"Range": "bytes=0-9"})

    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-length"] == str(len(VIDEO))
    assert head.headers["accept-ranges"] == "bytes"
    assert (
        ranged.status_code == 206 and ranged.headers["content-range"] == f"bytes 0-9/{len(VIDEO)}"
    )


# --- caching --------------------------------------------------------------------------------
def test_versioned_files_are_cached_for_a_year_others_are_revalidated(exp, anon):
    versioned = anon.get(url(exp, "ghost/v1.mp4"))
    plain = anon.get(url(exp))

    assert versioned.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert plain.headers["cache-control"] == "public, no-cache"


def test_a_matching_etag_gets_not_modified(exp, anon):
    first = anon.get(url(exp))

    again = anon.get(url(exp), headers={"If-None-Match": first.headers["etag"]})
    changed = anon.get(url(exp), headers={"If-None-Match": '"something else"'})

    assert again.status_code == 304 and again.content == b""
    assert again.headers["etag"] == first.headers["etag"]
    assert changed.status_code == 200


# --- not found and traversal ------------------------------------------------------------------
def test_missing_things_are_404(exp, anon):
    assert anon.get(url(exp, "trials/t1/nope.mp4")).status_code == 404
    assert anon.get(f"/api/media/{uuid4()}/trials/t1/playback_720.mp4").status_code == 404
    assert anon.get(url(exp, "trials/t1")).status_code == 404  # a directory
    assert anon.get(url(exp, "trials/t1/nope.mp4")).json()["code"] == "media_not_found"


@pytest.mark.parametrize(
    "path",
    [
        "%2e%2e/%2e%2e/etc/passwd",
        "trials/%2e%2e/%2e%2e/outside.txt",
        "..%2f..%2foutside.txt",
        "trials/t1/%2e%2e/%2e%2e/%2e%2e/outside.txt",
        ".hidden",
        "trials//t1/playback_720.mp4",
        "trials/t1/playback_720.mp4%00.png",
        "%5c..%5coutside.txt",
    ],
)
def test_path_traversal_is_a_plain_404(exp, anon, paths, path):
    (paths.root / "outside.txt").write_text("secret")

    response = anon.get(f"/api/media/{exp.id}/{path}")

    assert response.status_code == 404
    assert response.json().get("code") == "media_not_found"  # from our route, not the router
    assert "secret" not in response.text


def test_another_experiments_files_are_out_of_reach(exp, repo, anon, paths):
    other = repo.create_experiment(title="other")
    (paths.experiment_dir(other.id)).mkdir(parents=True)
    (paths.experiment_dir(other.id) / "private.txt").write_text("other's data")

    response = anon.get(f"/api/media/{exp.id}/%2e%2e/{other.id}/private.txt")

    assert response.status_code == 404


def test_symlinks_cannot_lead_out_of_the_folder(exp, anon, paths, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (paths.experiment_dir(exp.id) / "link.txt").symlink_to(outside)

    assert anon.get(url(exp, "link.txt")).status_code == 404


def test_uploads_and_caches_are_not_reachable(exp, anon, paths):
    (paths.root / "cache" / "llm").mkdir(parents=True)
    (paths.root / "cache" / "llm" / "abc.json").write_text("{}")

    assert anon.get(url(exp, "%2e%2e/%2e%2e/cache/llm/abc.json")).status_code == 404


# --- the Volume -----------------------------------------------------------------------------
class SyncingVolume:
    """A Volume where another container's file only shows up after reload()."""

    def __init__(self, appear):
        self.appear = appear
        self.reloads = 0

    def commit(self):
        pass

    def reload(self):
        self.reloads += 1
        if self.appear:
            self.appear()


def test_a_file_written_by_another_container_appears_after_a_reload(exp, db, settings, paths):
    late = paths.experiment_dir(exp.id) / "ghost" / "v2.mp4"
    volume = SyncingVolume(lambda: late.write_bytes(b"new ghost"))
    client = TestClient(create_app(settings=settings, db=db, volume=volume))

    first = client.get(url(exp, "ghost/v2.mp4"))
    later = client.get(url(exp, "ghost/v2.mp4"))

    assert first.status_code == 200 and first.content == b"new ghost"
    assert later.status_code == 200
    assert volume.reloads == 1  # present files never trigger a reload


def test_a_file_that_is_really_missing_reloads_at_most_once_a_second(exp, db, settings):
    volume = SyncingVolume(None)
    client = TestClient(create_app(settings=settings, db=db, volume=volume))

    codes = [client.get(url(exp, "ghost/nope.mp4")).status_code for _ in range(5)]

    assert codes == [404] * 5
    assert volume.reloads == 1


def test_a_failing_reload_does_not_turn_a_404_into_a_500(exp, db, settings):
    class Broken:
        def commit(self):
            pass

        def reload(self):
            raise RuntimeError("volume busy")

    client = TestClient(create_app(settings=settings, db=db, volume=Broken()))

    assert client.get(url(exp, "ghost/nope.mp4")).status_code == 404
