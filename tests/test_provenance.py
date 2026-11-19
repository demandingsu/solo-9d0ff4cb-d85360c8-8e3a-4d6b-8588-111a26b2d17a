"""Provenance index: block summaries, region queries, resumable builds."""

from __future__ import annotations

import csv
import io
import threading
import time
from collections import Counter
from urllib.parse import unquote

import pytest
from starlette.testclient import TestClient

from app import provenance
from app.main import create_app
from tests.conftest import mask_bytes, png_bytes, tile_upload


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _publish(client) -> dict:
    response = client.post("/publish", follow_redirects=False)
    assert response.status_code == 303
    versions = client.get("/versions").json()
    assert versions
    return versions[0]  # latest first


def _wait_build(client, version_id: str, timeout: float = 30.0) -> dict:
    # Wait until the build attempt leaves a status different from the one
    # before it started (a resumed build begins from a 'failed' state, so we
    # cannot just wait for "not building").
    previous = client.get("/provenance/status").json().get(version_id)
    deadline = time.time() + timeout
    while True:
        status = client.get("/provenance/status").json().get(version_id)
        if (
            status is not None
            and status != previous
            and status["state"] != "building"
        ):
            return status
        if time.time() > deadline:
            raise AssertionError(f"build did not finish: {status}")
        time.sleep(0.02)


def _build(client, version_id: str) -> dict:
    response = client.post(
        "/provenance/build", data={"version": version_id}, follow_redirects=False
    )
    assert response.status_code == 303
    return _wait_build(client, version_id)


def _query(client, version_id: str, rect, expected_status: int = 200):
    x0, y0, x1, y1 = rect
    response = client.get(
        "/provenance/query",
        params={
            "version": version_id,
            "x0": str(x0),
            "y0": str(y0),
            "x1": str(x1),
            "y1": str(y1),
        },
    )
    assert response.status_code == expected_status, response.text
    return response


def _sources_csv(client, version_id: str) -> str:
    response = client.get(f"/download/{version_id}/sources.csv")
    assert response.status_code == 200
    return response.content.decode("utf-8")


def _brute_force(csv_text: str, x0: int, y0: int, x1: int, y1: int) -> Counter:
    counts: Counter = Counter()
    for row in csv.reader(io.StringIO(csv_text)):
        if row[0] == "x":
            continue
        x, y = int(row[0]), int(row[1])
        if x0 <= x < x1 and y0 <= y < y1:
            counts[(row[2], int(row[3]) if row[3] else 0)] += 1
    return counts


def _assert_matches(result: dict, expected: Counter, rect) -> None:
    x0, y0, x1, y1 = rect
    actual = Counter({("", 0): result["no_source"]})
    for row in result["rows"]:
        actual[(row["tile_id"], row["revision"])] = row["pixels"]
    assert actual == expected
    assert result["total"] == (x1 - x0) * (y1 - y0)
    assert sum(actual.values()) == result["total"]


def _upload_files(client, tile_id, revision, x, y, width, height, cloud=()):
    files: dict = {}
    tile_upload(
        files,
        tile_id=tile_id,
        revision=revision,
        x=x,
        y=y,
        image=png_bytes(width, height, (10, 20, 30)),
        mask=mask_bytes(width, height, cloud),
    )
    response = client.post("/tiles", files=files, follow_redirects=False)
    assert response.status_code == 303
    return response


@pytest.fixture()
def layered_client(client, canvas, upload):
    """130x70 canvas: A (left), B (right, partly cloudy) over fallback C."""
    canvas(130, 70)
    upload(tile_id="C", revision=1, x=15, y=15, width=100, height=40,
           colour=(1, 1, 1))
    upload(tile_id="A", revision=1, x=0, y=0, width=60, height=70,
           colour=(2, 2, 2))
    upload(
        tile_id="B",
        revision=1,
        x=60,
        y=0,
        width=70,
        height=70,
        colour=(3, 3, 3),
        cloud={(0, 0), (5, 5), (69, 69), (10, 20)},
    )
    return client


# --------------------------------------------------------------------------- #
# Queries across blocks, edge blocks and no-source pixels
# --------------------------------------------------------------------------- #


def test_full_canvas_counts(layered_client):
    version = _publish(layered_client)
    assert _build(layered_client, version["id"])["state"] == "ready"
    result = _query(layered_client, version["id"], (0, 0, 130, 70)).json()
    by_tile = {row["tile_id"]: row["pixels"] for row in result["rows"]}
    assert by_tile == {"A": 60 * 70, "B": 70 * 70 - 4, "C": 1}
    # B's cloudy pixels without a clear fallback stay uncounted (no source).
    assert result["no_source"] == 3
    assert result["total"] == 130 * 70


def test_rects_across_blocks_match_brute_force(layered_client):
    version = _publish(layered_client)
    _build(layered_client, version["id"])
    csv_text = _sources_csv(layered_client, version["id"])
    rects = [
        (0, 0, 130, 70),    # whole canvas: every block fully inside
        (0, 0, 64, 64),     # exactly one interior block
        (64, 64, 130, 70),  # exactly the edge blocks (2x6 real size)
        (50, 60, 70, 68),   # crosses the x=64 and y=64 block boundaries
        (63, 63, 65, 65),   # 2x2 pixels touching four blocks
        (0, 63, 130, 65),   # strip across all block columns at y=64
        (128, 0, 130, 70),  # strip along the right edge blocks
        (60, 0, 70, 70),    # vertical strip crossing x=64
        (0, 0, 1, 1),       # single pixel, partial block
        (129, 69, 130, 70), # single no-source pixel (B cloud)
        (70, 20, 71, 21),   # single pixel sourced from fallback C
        (64, 0, 128, 64),   # exactly the middle interior block
    ]
    for rect in rects:
        result = _query(layered_client, version["id"], rect).json()
        _assert_matches(result, _brute_force(csv_text, *rect), rect)


def test_no_source_listed_separately(client, canvas, upload):
    canvas(70, 4)
    upload(tile_id="ONLY", revision=1, x=0, y=0, width=60, height=4)
    version = _publish(client)
    _build(client, version["id"])
    result = _query(client, version["id"], (0, 0, 70, 4)).json()
    assert result["rows"] == [
        {"tile_id": "ONLY", "revision": 1, "pixels": 240, "share": "85.71"}
    ]
    assert result["no_source"] == 40
    assert result["total"] == 280


# --------------------------------------------------------------------------- #
# Index lifecycle
# --------------------------------------------------------------------------- #


def test_query_requires_ready_index(client, canvas, upload):
    canvas(4, 4)
    upload(tile_id="A", revision=1, width=4, height=4)
    version = _publish(client)
    response = _query(client, version["id"], (0, 0, 4, 4), expected_status=409)
    assert "尚未就绪" in response.json()["detail"]
    # The HTML page reports the same instead of failing.
    page = client.get(
        "/",
        params={
            "prov_version": version["id"],
            "prov_x0": "0", "prov_y0": "0", "prov_x1": "4", "prov_y1": "4",
        },
    )
    assert page.status_code == 200
    assert "尚未就绪" in page.text


def test_rect_validation(client, canvas, upload):
    canvas(8, 8)
    upload(tile_id="A", revision=1, width=8, height=8)
    version = _publish(client)
    _build(client, version["id"])
    bad_rects = [
        (4, 0, 4, 8),    # empty
        (6, 0, 4, 8),    # x0 > x1
        (-1, 0, 4, 8),   # negative
        (0, 0, 9, 8),    # beyond canvas
        (0, 7, 8, 7),    # y0 == y1
    ]
    for rect in bad_rects:
        response = _query(client, version["id"], rect, expected_status=400)
        assert "detail" in response.json()
    assert _query(client, version["id"], (0, 0, 8, 8)).status_code == 200
    # Non-integer input and unknown versions are rejected too.
    assert client.get(
        "/provenance/query",
        params={"version": version["id"], "x0": "a", "y0": "0",
                "x1": "8", "y1": "8"},
    ).status_code == 400
    assert _query(client, "no-such-version", (0, 0, 1, 1),
                  expected_status=404).status_code == 404


def test_old_versions_can_be_backfilled(client, canvas, upload):
    canvas(8, 8)
    upload(tile_id="X", revision=1, width=8, height=8, colour=(9, 9, 9))
    first = _publish(client)
    upload(tile_id="Y", revision=1, x=4, y=0, width=4, height=8,
           colour=(8, 8, 8))
    second = _publish(client)

    # Build the older version after the newer one exists.
    assert _build(client, first["id"])["state"] == "ready"
    result = _query(client, first["id"], (0, 0, 8, 8)).json()
    assert {row["tile_id"] for row in result["rows"]} == {"X"}

    _build(client, second["id"])
    result = _query(client, second["id"], (0, 0, 8, 8)).json()
    by_tile = {row["tile_id"]: row["pixels"] for row in result["rows"]}
    assert by_tile == {"X": 32, "Y": 32}

    # The older index is untouched by newer builds.
    again = _query(client, first["id"], (0, 0, 8, 8)).json()
    assert {row["tile_id"] for row in again["rows"]} == {"X"}


def test_build_is_idempotent_and_unknown_version_rejected(client, canvas, upload):
    canvas(4, 4)
    upload(tile_id="A", revision=1, width=4, height=4)
    version = _publish(client)
    _build(client, version["id"])
    # Building a ready index is a no-op, not an error.
    response = client.post(
        "/provenance/build", data={"version": version["id"]},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "已就绪" in unquote(response.headers["location"])
    response = client.post(
        "/provenance/build", data={"version": "nope"}, follow_redirects=False
    )
    assert "err=" in response.headers["location"]


# --------------------------------------------------------------------------- #
# Resumable builds
# --------------------------------------------------------------------------- #


def test_failed_build_resumes_from_last_batch(client, canvas, upload, monkeypatch):
    canvas(200, 200)  # four 64-row batches
    upload(tile_id="D", revision=1, width=200, height=200)
    version = _publish(client)

    def fail_second_batch(batch_no: int) -> None:
        if batch_no == 2:
            raise RuntimeError("simulated disk error")

    monkeypatch.setattr(provenance, "batch_hook", fail_second_batch)
    status = _build(client, version["id"])
    assert status["state"] == "failed"
    assert status["rows_done"] == 128  # two committed 64-row batches
    assert "disk error" in status["error"]

    monkeypatch.setattr(provenance, "batch_hook", None)
    assert _build(client, version["id"])["state"] == "ready"

    csv_text = _sources_csv(client, version["id"])
    for rect in [(0, 0, 200, 200), (63, 127, 193, 200), (100, 100, 150, 150)]:
        result = _query(client, version["id"], rect).json()
        _assert_matches(result, _brute_force(csv_text, *rect), rect)


def test_restart_continues_interrupted_build(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    app_one = create_app(data_dir)
    with TestClient(app_one) as client_one:
        client_one.post("/canvas", data={"width": "150", "height": "150"})
        _upload_files(client_one, "T", 1, 0, 0, 150, 150)
        version = _publish(client_one)

        def fail_first_batch(batch_no: int) -> None:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(provenance, "batch_hook", fail_first_batch)
        status = _build(client_one, version["id"])
        assert status["state"] == "failed"
        assert status["rows_done"] == 64
        monkeypatch.setattr(provenance, "batch_hook", None)

        # Simulate a process that died mid-build: it never recorded the
        # failure, so the row still says "building" when the service stops.
        with client_one.app.state.db_lock as conn:
            conn.execute(
                "UPDATE provenance_indexes SET state = 'building' "
                "WHERE version_id = ?",
                (version["id"],),
            )

    # A fresh app instance on the same data dir reconciles the stale state
    # and resumes from the last committed batch instead of starting over.
    app_two = create_app(data_dir)
    with TestClient(app_two) as client_two:
        status = client_two.get("/provenance/status").json()[version["id"]]
        assert status["state"] == "failed"
        assert status["rows_done"] == 64
        assert "重启" in status["error"]

        assert _build(client_two, version["id"])["state"] == "ready"
        csv_text = _sources_csv(client_two, version["id"])
        for rect in [(0, 0, 150, 150), (64, 64, 150, 150), (10, 140, 149, 150)]:
            result = _query(client_two, version["id"], rect).json()
            _assert_matches(result, _brute_force(csv_text, *rect), rect)


def test_concurrent_build_requests_do_not_duplicate(client, canvas, upload, monkeypatch):
    canvas(200, 200)
    upload(tile_id="D", revision=1, width=200, height=200)
    version = _publish(client)

    entered = threading.Event()
    release = threading.Event()

    def pause_first_batch(batch_no: int) -> None:
        if batch_no == 1:
            entered.set()
            release.wait(5)

    monkeypatch.setattr(provenance, "batch_hook", pause_first_batch)
    try:
        client.post("/provenance/build", data={"version": version["id"]},
                    follow_redirects=False)
        assert entered.wait(5)
        # A second request while the build runs is reported, not spawned again.
        response = client.post(
            "/provenance/build", data={"version": version["id"]},
            follow_redirects=False,
        )
        assert "正在构建中" in unquote(response.headers["location"])
    finally:
        release.set()
        monkeypatch.setattr(provenance, "batch_hook", None)
    assert _wait_build(client, version["id"])["state"] == "ready"


# --------------------------------------------------------------------------- #
# Page integration
# --------------------------------------------------------------------------- #


def test_page_lists_versions_and_shows_results(layered_client):
    version = _publish(layered_client)
    page = layered_client.get("/")
    assert page.status_code == 200
    assert "区域来源统计" in page.text
    assert "构建索引" in page.text

    _build(layered_client, version["id"])
    page = layered_client.get(
        "/",
        params={
            "prov_version": version["id"],
            "prov_x0": "50", "prov_y0": "60",
            "prov_x1": "70", "prov_y1": "68",
        },
    )
    assert page.status_code == 200
    assert "（无来源）" in page.text
    assert ">A</td>" in page.text and ">B</td>" in page.text
    assert "[50, 70) × [60, 68)" in page.text
    # The index is now offered as ready.
    assert "就绪" in page.text
