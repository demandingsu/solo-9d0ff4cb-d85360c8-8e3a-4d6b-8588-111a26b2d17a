"""Per-version provenance index over a published ``sources.csv``.

The canvas is split into 64x64 pixel blocks starting at the top-left corner;
edge blocks keep their real width/height.  The index stores only summaries and
location info:

* ``provenance_blocks`` — per block, the pixels each ``(tile_id, revision)``
  contributed, plus the no-source count (``tile_id=''``, ``revision=0``);
* ``provenance_row_offsets`` — the byte offset of every pixel row's first CSV
  record, so a query that only partially covers a block can re-read exactly
  the CSV records it needs and count just the pixels inside the rectangle.

Builds commit after every batch of at most 64 pixel rows (one block row), so a
failed build or a service restart resumes from the last committed batch.  A
version can be queried only once its index is fully built (state ``ready``).
"""

from __future__ import annotations

import csv as csv_module
import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .mosaic import utc_now_iso

BLOCK_SIZE = 64
BATCH_PIXEL_ROWS = 64  # each build batch processes at most this many pixel rows

STATE_BUILDING = "building"
STATE_READY = "ready"
STATE_FAILED = "failed"

CSV_HEADER = b"x,y,tile_id,revision"

# Test hook: invoked with the 1-based batch number after each committed batch.
batch_hook: Callable[[int], None] | None = None


class ProvenanceError(RuntimeError):
    """The index or the CSV backing it is missing or inconsistent."""


class QueryRejected(ValueError):
    """The query cannot run; carries the HTTP status the API should use."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #


def _parse_line(line: bytes, cache: dict) -> tuple[int, int, str, int]:
    """Parse one ``x,y,tile_id,revision`` record.

    Tile ids never contain characters that need CSV quoting, so a plain split
    is exact; the csv module is the defensive fallback.  ``(tile, revision)``
    pairs repeat heavily, so they are decoded once and cached.
    """
    parts = line.rstrip(b"\r\n").split(b",")
    if len(parts) == 4 and b'"' not in line:
        x_b, y_b, tid_b, rev_b = parts
        key = (tid_b, rev_b)
        label = cache.get(key)
        if label is None:
            label = (tid_b.decode("utf-8"), int(rev_b) if rev_b else 0)
            cache[key] = label
        return int(x_b), int(y_b), label[0], label[1]
    try:
        row = next(csv_module.reader(io.StringIO(line.decode("utf-8"))))
    except Exception as exc:
        raise ProvenanceError(f"来源 CSV 行无法解析：{line[:80]!r}") from exc
    if len(row) != 4:
        raise ProvenanceError(f"来源 CSV 行字段数异常：{line[:80]!r}")
    revision = int(row[3]) if row[3] else 0
    return int(row[0]), int(row[1]), row[2], revision


def _verify_csv(csv_path: Path, expected_sha: str) -> None:
    if not csv_path.is_file():
        raise ProvenanceError("该版本的来源 CSV 文件缺失。")
    digest = hashlib.sha256()
    with open(csv_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ProvenanceError("来源 CSV 校验失败，无法构建索引。")


# --------------------------------------------------------------------------- #
# Status helpers
# --------------------------------------------------------------------------- #


def status_label(state: str | None, rows_done: int, height: int) -> str:
    if state is None:
        return "未构建"
    if state == STATE_READY:
        return "就绪"
    if state == STATE_BUILDING:
        percent = (rows_done * 100) // height if height else 0
        return f"构建中 {percent}%（{rows_done}/{height} 行）"
    return f"失败/中断于 {rows_done}/{height} 行，可继续构建"


def status_map(conn) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT version_id, state, rows_done, canvas_height, error, updated_at "
        "FROM provenance_indexes"
    ):
        out[row["version_id"]] = {
            "state": row["state"],
            "rows_done": row["rows_done"],
            "height": row["canvas_height"],
            "error": row["error"],
            "updated_at": row["updated_at"],
            "label": status_label(
                row["state"], row["rows_done"], row["canvas_height"]
            ),
        }
    return out


def reconcile_interrupted(conn) -> None:
    """Mark builds that were in flight when the service stopped as resumable."""
    conn.execute(
        "UPDATE provenance_indexes SET state = ?, error = ?, updated_at = ? "
        "WHERE state = ?",
        (
            STATE_FAILED,
            "构建被服务重启中断，点击“继续构建”可从上次进度继续。",
            utc_now_iso(),
            STATE_BUILDING,
        ),
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def run_build(
    db_lock,
    csv_path: Path,
    csv_sha256: str,
    version_id: str,
    width: int,
    height: int,
) -> None:
    """Build (or resume) the index; the outcome is recorded in the database."""
    try:
        _verify_csv(csv_path, csv_sha256)
        _build_batches(db_lock, csv_path, csv_sha256, version_id, width, height)
        with db_lock as conn:
            conn.execute(
                "UPDATE provenance_indexes SET state = ?, error = '', "
                "updated_at = ? WHERE version_id = ?",
                (STATE_READY, utc_now_iso(), version_id),
            )
    except Exception as exc:  # noqa: BLE001 - recorded so the UI can resume
        message = str(exc).strip().splitlines()[0][:200] or "未知错误"
        try:
            with db_lock as conn:
                conn.execute(
                    "UPDATE provenance_indexes SET state = ?, error = ?, "
                    "updated_at = ? WHERE version_id = ?",
                    (STATE_FAILED, message, utc_now_iso(), version_id),
                )
        except Exception:
            pass


def _prepare_build(
    db_lock, csv_sha256: str, version_id: str, width: int, height: int
) -> tuple[int, int | None]:
    """Create or resume the index row; returns (rows_done, seek_offset)."""
    with db_lock as conn:
        row = conn.execute(
            "SELECT rows_done, csv_sha256 FROM provenance_indexes "
            "WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if row is not None and row["csv_sha256"] != csv_sha256:
            # The backing file changed (should not happen for immutable
            # versions): discard the stale progress and rebuild from zero.
            conn.execute(
                "DELETE FROM provenance_blocks WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "DELETE FROM provenance_row_offsets WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "DELETE FROM provenance_indexes WHERE version_id = ?",
                (version_id,),
            )
            row = None
        if row is None:
            conn.execute(
                "DELETE FROM provenance_blocks WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "DELETE FROM provenance_row_offsets WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "INSERT INTO provenance_indexes(version_id, state, canvas_width, "
                "canvas_height, rows_done, csv_sha256, error, updated_at) "
                "VALUES (?, ?, ?, ?, 0, ?, '', ?)",
                (
                    version_id,
                    STATE_BUILDING,
                    width,
                    height,
                    csv_sha256,
                    utc_now_iso(),
                ),
            )
            return 0, None
        rows_done = row["rows_done"]
        # Defensive: nothing beyond the last committed batch may survive.
        conn.execute(
            "DELETE FROM provenance_row_offsets WHERE version_id = ? AND y > ?",
            (version_id, rows_done),
        )
        conn.execute(
            "DELETE FROM provenance_blocks WHERE version_id = ? AND by >= ?",
            (version_id, (rows_done + BLOCK_SIZE - 1) // BLOCK_SIZE),
        )
        conn.execute(
            "UPDATE provenance_indexes SET state = ?, error = '', updated_at = ? "
            "WHERE version_id = ?",
            (STATE_BUILDING, utc_now_iso(), version_id),
        )
        if rows_done == 0:
            return 0, None
        offset_row = conn.execute(
            "SELECT byte_offset FROM provenance_row_offsets "
            "WHERE version_id = ? AND y = ?",
            (version_id, rows_done),
        ).fetchone()
        if offset_row is None:
            # The resume cursor and the stored offsets disagree (index
            # corruption).  The index is derived data, so restart from zero
            # instead of getting stuck.
            conn.execute(
                "DELETE FROM provenance_blocks WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "DELETE FROM provenance_row_offsets WHERE version_id = ?",
                (version_id,),
            )
            conn.execute(
                "UPDATE provenance_indexes SET rows_done = 0 WHERE version_id = ?",
                (version_id,),
            )
            return 0, None
        return rows_done, offset_row["byte_offset"]


def _build_batches(
    db_lock,
    csv_path: Path,
    csv_sha256: str,
    version_id: str,
    width: int,
    height: int,
) -> None:
    rows_done, start_offset = _prepare_build(
        db_lock, csv_sha256, version_id, width, height
    )
    cache: dict = {}
    with open(csv_path, "rb") as handle:
        if start_offset is None:
            header = handle.readline()
            if header.rstrip(b"\r\n") != CSV_HEADER:
                raise ProvenanceError("来源 CSV 表头不符。")
        else:
            handle.seek(start_offset)
        y = rows_done
        batch_no = 0
        while y < height:
            # One batch = one block row: at most BATCH_PIXEL_ROWS pixel rows.
            batch_end = min(
                y + (BATCH_PIXEL_ROWS - y % BATCH_PIXEL_ROWS), height
            )
            offsets: dict[int, int] = {}
            block_counts: Counter[tuple[int, str, int]] = Counter()
            for row_y in range(y, batch_end):
                offsets[row_y] = handle.tell()
                for expect_x in range(width):
                    line = handle.readline()
                    if not line:
                        raise ProvenanceError(
                            f"来源 CSV 在第 {row_y} 行附近提前结束。"
                        )
                    px, py, tile_id, revision = _parse_line(line, cache)
                    if px != expect_x or py != row_y:
                        raise ProvenanceError(
                            f"来源 CSV 坐标异常：期望 {expect_x},{row_y}，"
                            f"实际 {px},{py}。"
                        )
                    block_counts[(px // BLOCK_SIZE, tile_id, revision)] += 1
            # Resume point: the first record of the next pixel row.
            offsets[batch_end] = handle.tell()
            by = y // BLOCK_SIZE
            with db_lock as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO provenance_row_offsets"
                    "(version_id, y, byte_offset) VALUES (?, ?, ?)",
                    [
                        (version_id, offset_y, offset)
                        for offset_y, offset in offsets.items()
                    ],
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO provenance_blocks"
                    "(version_id, bx, by, tile_id, revision, pixels) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (version_id, bx, by, tile_id, revision, pixels)
                        for (bx, tile_id, revision), pixels in sorted(
                            block_counts.items()
                        )
                    ],
                )
                conn.execute(
                    "UPDATE provenance_indexes SET rows_done = ?, updated_at = ? "
                    "WHERE version_id = ?",
                    (batch_end, utc_now_iso(), version_id),
                )
            batch_no += 1
            if batch_hook is not None:
                batch_hook(batch_no)
            y = batch_end
        if handle.read(1):
            raise ProvenanceError("来源 CSV 行数多于画幅高度。")


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #


def query_region(
    conn,
    csv_path: Path,
    version_id: str,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> dict[str, Any]:
    """Count contributed pixels per source inside [x0,x1) x [y0,y1).

    Blocks fully inside the rectangle are answered from the stored summaries;
    only partially covered blocks re-read their CSV records, counting just the
    pixels inside the rectangle.
    """
    counts: Counter[tuple[str, int]] = Counter()
    partial: dict[int, list[tuple[int, int]]] = {}
    for by in range(y0 // BLOCK_SIZE, (y1 - 1) // BLOCK_SIZE + 1):
        by0 = by * BLOCK_SIZE
        by1 = min(by0 + BLOCK_SIZE, height)
        for bx in range(x0 // BLOCK_SIZE, (x1 - 1) // BLOCK_SIZE + 1):
            bx0 = bx * BLOCK_SIZE
            bx1 = min(bx0 + BLOCK_SIZE, width)
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            if (ix0, iy0, ix1, iy1) == (bx0, by0, bx1, by1):
                for row in conn.execute(
                    "SELECT tile_id, revision, pixels FROM provenance_blocks "
                    "WHERE version_id = ? AND bx = ? AND by = ?",
                    (version_id, bx, by),
                ):
                    counts[(row["tile_id"], row["revision"])] += row["pixels"]
            else:
                for yy in range(iy0, iy1):
                    partial.setdefault(yy, []).append((ix0, ix1))
    if partial:
        _count_partial_rows(conn, csv_path, version_id, partial, counts)
    total = (x1 - x0) * (y1 - y0)
    counted = sum(counts.values())
    if counted != total:
        raise ProvenanceError(
            f"索引统计 {counted} 与矩形面积 {total} 不符，请重建索引。"
        )
    no_source = counts.pop(("", 0), 0)
    rows = [
        {
            "tile_id": tile_id,
            "revision": revision,
            "pixels": pixels,
            "share": f"{pixels * 100 / total:.2f}",
        }
        for (tile_id, revision), pixels in sorted(
            counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
    ]
    return {
        "total": total,
        "no_source": no_source,
        "no_source_share": f"{no_source * 100 / total:.2f}",
        "rows": rows,
    }


def _count_partial_rows(
    conn,
    csv_path: Path,
    version_id: str,
    partial: dict[int, list[tuple[int, int]]],
    counts: Counter,
) -> None:
    """Re-read CSV records for partially covered blocks.

    ``partial`` maps a pixel row to the disjoint x-ranges (half-open) that lie
    inside the query rectangle.  Only pixels inside those ranges are counted.
    """
    wanted = sorted(partial)
    offsets = {
        row["y"]: row["byte_offset"]
        for row in conn.execute(
            "SELECT y, byte_offset FROM provenance_row_offsets "
            "WHERE version_id = ? AND y BETWEEN ? AND ?",
            (version_id, wanted[0], wanted[-1]),
        )
    }
    cache: dict = {}
    with open(csv_path, "rb") as handle:
        for yy in wanted:
            offset = offsets.get(yy)
            if offset is None:
                raise ProvenanceError("索引缺少行偏移，请重建索引。")
            ranges = partial[yy]
            last_x = max(hi for _, hi in ranges)
            handle.seek(offset)
            for x in range(last_x):
                line = handle.readline()
                if not line:
                    raise ProvenanceError("来源 CSV 读取提前结束。")
                if not any(lo <= x < hi for lo, hi in ranges):
                    continue
                px, py, tile_id, revision = _parse_line(line, cache)
                if px != x or py != yy:
                    raise ProvenanceError("来源 CSV 与索引不一致。")
                counts[(tile_id, revision)] += 1
