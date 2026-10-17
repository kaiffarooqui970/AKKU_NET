"""Tests for Stage 1 frame extraction (synthetic data only)."""
import cv2
import numpy as np
import pytest

from akkuwatch.frames import (
    assign_splits,
    dhash,
    extract_session,
    find_sessions,
    hamming,
    laplacian_variance,
)
from akkuwatch.synthetic import make_synthetic_video

FE_CFG = {
    "every_n_frames": 1,
    "jpeg_quality": 95,
    "blur_threshold": 100.0,
    "blur_measure_long_side": 640,
    "hash_size": 16,
    "hamming_threshold": 10,
}


def random_blocks(seed, size=(240, 320)):
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(size[0] // 16, size[1] // 16, 3), dtype=np.uint8)
    return cv2.resize(blocks, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- blur

def test_blurred_image_scores_lower():
    sharp = random_blocks(0)
    blurred = cv2.GaussianBlur(sharp, (51, 51), 0)
    assert laplacian_variance(blurred) < 100 < laplacian_variance(sharp)


def test_blur_score_is_roughly_resolution_independent():
    img = random_blocks(0)
    big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    a, b = laplacian_variance(img), laplacian_variance(big)
    assert abs(a - b) / a < 0.5


# ---------------------------------------------------------------- hashing

def test_identical_images_have_identical_hash():
    img = random_blocks(1)
    assert hamming(dhash(img), dhash(img.copy())) == 0


def test_small_change_keeps_hash_close_and_new_scene_is_far():
    img = random_blocks(1)
    noisy = np.clip(img.astype(int) + np.random.default_rng(9).integers(-5, 6, img.shape), 0, 255).astype(np.uint8)
    assert hamming(dhash(img), dhash(noisy)) <= 10
    assert hamming(dhash(img), dhash(random_blocks(2))) > 60


# ---------------------------------------------------------------- splits

SESSIONS = ["s1", "s2", "s3", "s4", "s5"]


def test_split_assignment_unlisted_sessions_go_to_train():
    assignment, warnings = assign_splits(
        SESSIONS, {"val_sessions": ["s3"], "test_sessions": ["s4"], "shift_test_sessions": ["s5"]}
    )
    assert assignment == {"s1": "train", "s2": "train", "s3": "val", "s4": "test", "s5": "shift_test"}
    assert warnings == []


def test_shift_test_session_is_required():
    with pytest.raises(ValueError, match="shift_test"):
        assign_splits(SESSIONS, {"val_sessions": ["s3"], "test_sessions": ["s4"], "shift_test_sessions": []})


def test_session_in_two_splits_is_rejected():
    with pytest.raises(ValueError, match="both"):
        assign_splits(SESSIONS, {"val_sessions": ["s3"], "test_sessions": ["s3"], "shift_test_sessions": ["s5"]})


def test_typo_in_session_name_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        assign_splits(SESSIONS, {"shift_test_sessions": ["s_5"]})


def test_no_training_session_left_is_rejected():
    with pytest.raises(ValueError, match="train"):
        assign_splits(["a", "b"], {"val_sessions": ["a"], "shift_test_sessions": ["b"]})


def test_missing_val_and_test_give_warnings():
    _, warnings = assign_splits(SESSIONS, {"shift_test_sessions": ["s5"]})
    assert len(warnings) == 2


# ---------------------------------------------------------------- end to end

def test_extract_session_removes_blur_and_duplicates(tmp_path):
    # 4 distinct scenes x 5 identical frames, then 1 blurred scene x 5 frames.
    video = tmp_path / "raw" / "sess" / "clip.mp4"
    make_synthetic_video(video, n_scenes=4, frames_per_scene=5, n_blurry_scenes=1, seed=3)

    stats, rows = extract_session("sess", "train", [video], tmp_path / "images", FE_CFG)

    assert stats.sampled == 25
    assert stats.blurry == 5
    assert stats.kept == 4  # one frame per distinct scene
    assert stats.duplicates == 16
    saved = sorted((tmp_path / "images" / "train").glob("*.jpg"))
    assert len(saved) == 4 and len(rows) == 4
    assert all(p.name.startswith("sess__clip__f") for p in saved)


def test_every_n_frames_sampling(tmp_path):
    video = tmp_path / "clip.mp4"
    make_synthetic_video(video, n_scenes=4, frames_per_scene=5, n_blurry_scenes=0)
    stats, _ = extract_session("s", "train", [video], tmp_path / "images", {**FE_CFG, "every_n_frames": 5})
    # frames 0, 5, 10, 15 -> exactly one sample per scene, all distinct
    assert stats.sampled == 4 and stats.kept == 4


def test_duplicates_detected_across_videos_in_same_session(tmp_path):
    # Two videos with the same seed = same scenes: the second adds nothing new.
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for v in (v1, v2):
        make_synthetic_video(v, n_scenes=3, frames_per_scene=2, n_blurry_scenes=0, seed=7)
    stats, _ = extract_session("s", "train", [v1, v2], tmp_path / "images", FE_CFG)
    assert stats.kept == 3


def test_find_sessions_ignores_non_videos(tmp_path):
    make_synthetic_video(tmp_path / "session_01" / "clip.mp4", n_scenes=1, frames_per_scene=1)
    (tmp_path / "session_01" / "notes.txt").write_text("hi")
    (tmp_path / "empty_session").mkdir()
    sessions = find_sessions(tmp_path, [".mp4", ".mov"])
    assert list(sessions) == ["session_01"]
    assert [v.name for v in sessions["session_01"]] == ["clip.mp4"]
