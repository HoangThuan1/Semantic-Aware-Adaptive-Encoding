import json
from types import SimpleNamespace

import pandas as pd

from vcu_surrogate import legacy_metrics
from vcu_surrogate.common import sha256, stable_hash, write_json


def test_reuses_only_identical_encoded_checksum_and_detections(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_metrics, "ROOT", tmp_path)
    old, current = tmp_path / "artifacts", tmp_path / "current"
    monkeypatch.setattr(legacy_metrics, "project_path", lambda config, group: current)
    for root in (old, current):
        for folder in ("reference", "metadata", "metrics/v"):
            (root / folder).mkdir(parents=True)
        write_json(root / "reference/v.json", {"source_sha256": "source", "video": {"fps": 30},
                                               "keep_frames": 30, "output_sha256": "old-reference"})
        pd.DataFrame({"frame_idx": [0], "x1": [1.]}).to_parquet(root / "metadata/v.detections.parquet")
    encoded = tmp_path / "encode.mp4"
    encoded.write_bytes(b"same bitstream")
    job = {"encoded_sha256": sha256(encoded), "reference_sha256": "old-reference",
           "metrics": ["vmaf", "psnr_y", "float_ssim", "roi_psnr", "roi_ssim"]}
    write_json(old / "metrics/v/action_00.json", {**job, "job_hash": stable_hash(job)})
    pd.DataFrame({"frame_idx": range(30)}).to_parquet(old / "metrics/v/action_00.frames.parquet")
    pd.DataFrame([dict(gop_idx=0, action_id=0, video_id="v", vmaf=80., psnr=35., ssim=.9,
                       actual_bitrate_mbps=1.)]).to_parquet(old / "metrics/v/action_00.gops.parquet")
    config = {"paths": {"artifacts": "current"}}
    reference = SimpleNamespace(video_id="v", frame_count=30, gop_count=1,
                                reference_sha256="new-container-same-pixels", dataset_sha256="new-dataset")
    encodes = pd.DataFrame([dict(action_id=0, encoded_sha256="different", encoded_path="encode.mp4")])
    assert legacy_metrics.reuse_legacy_metrics(config, reference, encodes) == 0
    encodes.loc[0, "encoded_sha256"] = sha256(encoded)
    assert legacy_metrics.reuse_legacy_metrics(config, reference, encodes) == 1
    result = json.loads((current / "metrics/v/action_00.json").read_text())
    assert result["dataset_sha256"] == "new-dataset"
    assert result["legacy_reference_sha256"] == "old-reference"
    pd.DataFrame({"frame_idx": [0], "x1": [2.]}).to_parquet(current / "metadata/v.detections.parquet")
    assert legacy_metrics.reuse_legacy_metrics(config, reference, encodes) == 0
