from uuid import uuid4

import pytest

from ghostline.store.volume import (
    NullVolume,
    UnsafePath,
    VolumePaths,
    is_content_hashed,
    safe_name,
)

EXP, TRIAL, FIT, UPLOAD = (uuid4() for _ in range(4))


def test_layout_matches_the_plan(paths):
    root = paths.root
    assert (
        paths.original(EXP, TRIAL, "mov") == root / f"experiments/{EXP}/trials/{TRIAL}/original.mov"
    )
    assert paths.trial_file(EXP, TRIAL, "master.mp4").name == "master.mp4"
    assert paths.sam_events(EXP, TRIAL, "bob") == (
        root / f"experiments/{EXP}/trials/{TRIAL}/sam/bob.events.jsonl"
    )
    assert paths.calib_file(EXP, TRIAL, "overlay.png").parent.name == "calib"
    assert paths.fit_dir(EXP, FIT) == root / f"experiments/{EXP}/fits/{FIT}"
    assert paths.ghost_video(EXP, 2) == root / f"experiments/{EXP}/ghost/v2.mp4"
    assert paths.ghost_track(EXP, 2).name == "v2_track.json"
    assert paths.ghost_slow(EXP, 2).name == "v2_slow.mp4"
    assert paths.llm_cache("abc123") == root / "cache/llm/abc123.json"
    assert paths.sam_cache("abc123") == root / "cache/sam/abc123.events.jsonl"
    assert paths.upload_part(UPLOAD, 3) == root / f"uploads/{UPLOAD}/3.part"
    assert paths.backups_dir() == root / "backups"


def test_rel_is_relative_to_the_experiment_folder(paths):
    path = paths.trial_file(EXP, TRIAL, "master.mp4")

    assert paths.rel(EXP, path) == f"trials/{TRIAL}/master.mp4"
    assert paths.resolve_media(EXP, paths.rel(EXP, path)) == path.resolve()


@pytest.mark.parametrize(
    "name", ["..", ".hidden", "a/b", "a\\b", "", "x" * 200, "a b", "../x", "é"]
)
def test_safe_name_rejects_anything_that_is_not_one_plain_component(name):
    with pytest.raises(UnsafePath):
        safe_name(name)


def test_builders_refuse_unsafe_ids(paths):
    with pytest.raises(UnsafePath):
        paths.experiment_dir("../../etc")
    with pytest.raises(UnsafePath):
        paths.sam_events(EXP, TRIAL, "../escape")
    with pytest.raises(UnsafePath):
        paths.original(EXP, TRIAL, "mp4/../../x")


@pytest.mark.parametrize(
    "relative",
    [
        "../other/master.mp4",
        "trials/../../x",
        "/etc/passwd",
        "trials//master.mp4",
        "trials/./master.mp4",
        ".secret",
        "trials/.hidden/x",
        "a\\b",
        "a\x00b",
        "",
    ],
)
def test_resolve_media_rejects_traversal(paths, relative):
    with pytest.raises(UnsafePath):
        paths.resolve_media(EXP, relative)


def test_resolve_media_does_not_follow_symlinks_out_of_the_experiment(paths, tmp_path):
    folder = paths.experiment_dir(EXP)
    folder.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (folder / "link.txt").symlink_to(outside)

    with pytest.raises(UnsafePath):
        paths.resolve_media(EXP, "link.txt")


@pytest.mark.parametrize(
    "name, hashed",
    [
        ("ghost/v1.mp4", True),
        ("ghost/v12_track.json", True),
        ("ghost/v2_slow.mp4", True),
        ("trials/t/abcdef0123456789.png", True),
        ("trials/t/master.mp4", False),
        ("trials/t/playback_720.mp4", False),
        ("trials/t/original.mov", False),
        ("fits/f/fit.png", False),
        ("trials/t/dead.mp4", False),
    ],
)
def test_content_hashed_names(name, hashed):
    assert is_content_hashed(name) is hashed


def test_null_volume_does_nothing():
    NullVolume().commit()
    NullVolume().reload()


def test_modal_volume_forwards_to_the_volume():
    from ghostline.store.volume import ModalVolume

    calls = []

    class Fake:
        def commit(self):
            calls.append("commit")

        def reload(self):
            calls.append("reload")

    volume = ModalVolume(Fake())
    volume.commit()
    volume.reload()

    assert calls == ["commit", "reload"]


def test_volume_paths_accept_strings():
    assert VolumePaths("/data").root.as_posix() == "/data"
