"""Housekeeping for the Volume (build plan Part 28).

The purge of finished experiments' media after 14 days (non-example, non-benchmark) is a policy
call about what a viewer should still see afterwards, so it is not done here. This removes the
chunks of uploads that were started and never completed, which can be up to 1 GiB each.
"""

import logging

from ghostline.store import uploads
from ghostline.store.repo import Repo
from ghostline.store.volume import VolumePaths, VolumeSync

log = logging.getLogger("ghostline.store")


def purge_stale_uploads(
    repo: Repo, paths: VolumePaths, volume: VolumeSync, *, older_than_hours: float
) -> int:
    """Delete incomplete uploads older than the cutoff: their chunk files and their rows.
    Returns how many were removed."""
    stale = repo.stale_open_uploads(older_than_hours)
    for upload in stale:
        uploads.remove_upload_dir(paths, upload.id)
        repo.delete_upload(upload.id)
    if stale:
        volume.commit()
        log.info("purged stale uploads", extra={"count": len(stale)})
    return len(stale)
