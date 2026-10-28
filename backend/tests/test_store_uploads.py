import hashlib
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ghostline.store import uploads
from ghostline.store.records import Upload


def make_upload(size: int, chunk: int = 10) -> Upload:
    return Upload(
        id=uuid4(),
        experiment_id=uuid4(),
        role="fit",
        filename="clip.MOV",
        size_bytes=size,
        chunk_size=chunk,
        n_chunks=uploads.n_chunks_for(size, chunk),
        status="open",
        created_at=datetime.now(UTC),
    )


def store_all(paths, upload: Upload, data: bytes) -> None:
    for n in range(upload.n_chunks):
        piece = data[n * upload.chunk_size : (n + 1) * upload.chunk_size]
        uploads.write_chunk(paths, upload.id, n, piece)


@pytest.mark.parametrize(
    "size, chunk, expected", [(1, 10, 1), (10, 10, 1), (11, 10, 2), (100, 10, 10), (101, 10, 11)]
)
def test_chunk_count_rounds_up(size, chunk, expected):
    assert uploads.n_chunks_for(size, chunk) == expected


def test_every_chunk_is_full_size_except_the_last():
    upload = make_upload(25, 10)

    assert [uploads.expected_chunk_size(upload, n) for n in range(3)] == [10, 10, 5]
    assert uploads.expected_chunk_size(make_upload(20, 10), 1) == 10


@pytest.mark.parametrize(
    "filename, ext",
    [
        ("clip.MOV", "mov"),
        ("a.mp4", "mp4"),
        ("weird.name.webm", "webm"),
        ("noext", "bin"),
        ("script.sh", "bin"),
        ("../../etc/passwd", "bin"),
        ("x.mp4/../../evil", "bin"),
        ("a.exe", "bin"),
    ],
)
def test_only_video_extensions_survive(filename, ext):
    assert uploads.extension_for(filename) == ext


def test_assemble_joins_chunks_and_hashes(paths, tmp_path):
    data = os.urandom(95)
    upload = make_upload(len(data))
    store_all(paths, upload, data)
    destination = tmp_path / "out" / "original.mov"

    digest, size = uploads.assemble(paths, upload, destination)

    assert destination.read_bytes() == data
    assert size == 95
    assert digest == hashlib.sha256(data).hexdigest()
    assert list(destination.parent.iterdir()) == [destination]  # no temp file left behind


def test_chunks_may_arrive_in_any_order_and_be_repeated(paths, tmp_path):
    data = b"abcdefghijklmnopqrstuvwxyz"
    upload = make_upload(len(data), 10)

    for n in (2, 0, 1, 1, 0):
        uploads.write_chunk(paths, upload.id, n, data[n * 10 : (n + 1) * 10])
    digest, _ = uploads.assemble(paths, upload, tmp_path / "o.bin")

    assert digest == hashlib.sha256(data).hexdigest()


def test_rewriting_a_chunk_replaces_it(paths, tmp_path):
    upload = make_upload(10, 10)
    uploads.write_chunk(paths, upload.id, 0, b"x" * 10)
    uploads.write_chunk(paths, upload.id, 0, b"y" * 10)

    uploads.assemble(paths, upload, tmp_path / "o.bin")

    assert (tmp_path / "o.bin").read_bytes() == b"y" * 10
    assert [p.name for p in paths.upload_dir(upload.id).iterdir()] == ["0.part"]


def test_missing_and_short_chunks_are_reported(paths, tmp_path):
    data = os.urandom(40)
    upload = make_upload(len(data), 10)
    uploads.write_chunk(paths, upload.id, 0, data[:10])
    uploads.write_chunk(paths, upload.id, 2, data[20:25])  # truncated: should be 10 bytes
    uploads.write_chunk(paths, upload.id, 3, data[30:40])

    assert uploads.missing_chunks(paths, upload) == [1, 2]
    with pytest.raises(uploads.UploadIncomplete) as exc:
        uploads.assemble(paths, upload, tmp_path / "o.bin")
    assert exc.value.missing == [1, 2]
    assert not (tmp_path / "o.bin").exists()


def test_failed_assembly_leaves_no_partial_file(paths, tmp_path):
    upload = make_upload(30, 10)
    store_all(paths, upload, b"z" * 30)
    destination = tmp_path / "dest" / "o.bin"
    paths.upload_part(upload.id, 2).unlink()
    paths.upload_part(upload.id, 2).mkdir()  # a directory where a chunk should be

    with pytest.raises(uploads.UploadIncomplete):
        uploads.assemble(paths, upload, destination)

    assert not destination.exists()


def test_remove_upload_dir(paths):
    upload = make_upload(10)
    store_all(paths, upload, b"x" * 10)

    uploads.remove_upload_dir(paths, upload.id)
    uploads.remove_upload_dir(paths, upload.id)  # already gone is fine

    assert not paths.upload_dir(upload.id).exists()


def test_stale_uploads_are_purged_with_their_chunks(repo, paths, db):
    from ghostline.store.retention import purge_stale_uploads
    from ghostline.store.volume import NullVolume

    exp = repo.create_experiment(title="t")

    def start(age_hours: float):
        upload = repo.create_upload(
            experiment_id=exp.id,
            role="fit",
            filename="a.mov",
            size_bytes=20,
            chunk_size=10,
            n_chunks=2,
        )
        uploads.write_chunk(paths, upload.id, 0, b"x" * 10)
        db.execute(
            "update uploads set created_at = now() - make_interval(hours => %s) where id = %s",
            (age_hours, upload.id),
        )
        return upload

    old, fresh, done = start(30), start(1), start(48)
    repo.mark_upload_complete(done.id, repo.create_trial(experiment_id=exp.id, role="fit").id)

    removed = purge_stale_uploads(repo, paths, NullVolume(), older_than_hours=24)

    assert removed == 1
    assert repo.get_upload(old.id) is None and not paths.upload_dir(old.id).exists()
    assert repo.get_upload(fresh.id) is not None and paths.upload_dir(fresh.id).exists()
    assert repo.get_upload(done.id) is not None  # completed uploads are not touched
    assert purge_stale_uploads(repo, paths, NullVolume(), older_than_hours=24) == 0
