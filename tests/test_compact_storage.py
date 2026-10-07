from __future__ import annotations

import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np

from vcu_surrogate import benchmark
from vcu_surrogate.common import sha256, write_json


def compact_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_grid_compact.py"
    spec = importlib.util.spec_from_file_location("compact_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_reference_can_cross_process_boundary():
    compact = compact_module()
    record = next(pd.DataFrame([{"video_id": "v", "frame_count": 30}]).itertuples(index=False))
    with ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(getattr, compact.metric_reference(record), "video_id").result() == "v"


@pytest.mark.parametrize("shape,window", [((3, 3, 3), 3), ((5, 9, 3), 5),
    ((7, 7, 3), 7), ((32, 45, 3), 7), ((256, 400, 3), 7)])
def test_native_roi_ssim_matches_skimage(shape, window):
    from skimage.metrics import structural_similarity
    from vcu_surrogate.metrics import _roi_ssim
    rng = np.random.default_rng(47)
    random = rng.integers(0, 256, shape, dtype=np.uint8)
    pairs = [(random, rng.integers(0, 256, shape, dtype=np.uint8)), (random, random),
             (np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8)),
             (np.full(shape, 127, np.uint8), np.full(shape, 128, np.uint8))]
    for reference, distorted in pairs:
        expected = structural_similarity(reference, distorted, channel_axis=2,
                                         data_range=255, win_size=window)
        assert _roi_ssim(reference, distorted, window) == pytest.approx(expected, abs=1e-10, rel=0)


def test_all_videos_scheduled_without_disk_cap():
    compact = compact_module()
    references = [SimpleNamespace(video_id=f'bdd_{i}', reference_size_bytes=200_000_000_000,
                                  gop_count=40) for i in range(30)]
    selected, reserved = compact.choose_batch(references, {}, required=['bdd_29'])
    assert len(selected) == len(references)
    assert selected[0] == 'bdd_29'
    assert reserved is None
    assert selected == compact.choose_batch(references, {}, required=['bdd_29'])[0]


def test_resume_shared_cache_rejects_changed_source_or_corrupt_pixels(tmp_path):
    compact = compact_module()
    reference = SimpleNamespace(video_id="video", reference_sha256="source", frame_count=1)
    raw, bgr = tmp_path / "ref.yuv", tmp_path / "ref.npy"
    raw.write_bytes(bytes(3110400))
    np.save(bgr, np.zeros((1, 1080, 1920, 3), dtype=np.uint8))
    receipt = {"video_id": "video", "reference_sha256": "source", "frame_count": 1,
               "yuv_sha256": sha256(raw), "bgr_sha256": sha256(bgr)}
    assert compact.checked_reference_cache(reference, raw, bgr, receipt)
    assert not compact.checked_reference_cache(reference, raw, bgr, {**receipt, "reference_sha256": "other"})
    with bgr.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"\x01")
    assert not compact.checked_reference_cache(reference, raw, bgr, receipt)


def test_verified_encode_resume_avoids_decode_and_rejects_corruption(tmp_path, monkeypatch):
    from vcu_surrogate import encoding
    from vcu_surrogate.common import stable_hash
    monkeypatch.setattr(encoding, "ROOT", tmp_path)
    monkeypatch.setattr(encoding, "dataset_hash", lambda config: "dataset")
    monkeypatch.setattr(encoding, "count_frames", lambda path: pytest.fail("verified resume must not decode"))
    config = {"encoder": {"roi_delta_qp": [0]}, "video": {"gop_frames": 30}}
    video = SimpleNamespace(video_id="v", reference_sha256="reference", frame_count=30, gop_count=1)
    output = tmp_path / "action_00_b1_q0.mp4"
    output.write_bytes(b"checked encoded stream")
    job = {"dataset_sha256": "dataset", "video_id": "v", "reference_sha256": "reference",
           "roi_sha256": "roi", "action_id": 0, "bitrate_mbps": 1.0, "delta_qp": 0,
           "encoder": config["encoder"], "gop_frames": 30}
    write_json(output.with_suffix(".json"), {**job, "job_hash": stable_hash(job), "output_sha256": sha256(output)})
    row = encoding._encode_one(video, tmp_path / "roi.csv", "roi", tmp_path, 0, 0, 1, 0, config)
    assert row["encoded_sha256"] == sha256(output)
    output.write_bytes(b"corrupt encoded stream")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        encoding._encode_one(video, tmp_path / "roi.csv", "roi", tmp_path, 0, 0, 1, 0, config)


def test_release_requires_all_actions_and_verified_cache(tmp_path, monkeypatch):
    compact = compact_module()
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    artifacts, reports = tmp_path / "artifacts", tmp_path / "benchmark"
    reference_path = artifacts / "reference/v.mkv"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes(b"lossless original")
    checksum = sha256(reference_path)
    ref = SimpleNamespace(video_id="v", reference_path="artifacts/reference/v.mkv",
                          reference_sha256=checksum, frame_count=30, gop_count=1)
    encodes = artifacts / "encodes/v/manifest.parquet"
    encodes.parent.mkdir(parents=True)
    pd.DataFrame({"action_id": range(31)}).to_parquet(encodes)
    with pytest.raises(RuntimeError, match="incomplete encode"):
        compact.release_reference(ref, artifacts, reports, "dataset")
    assert reference_path.exists()
    pd.DataFrame({"action_id": range(32)}).to_parquet(encodes)
    for action in range(32):
        metric = artifacts / "metrics/v" / f"action_{action:02d}"
        metric.parent.mkdir(parents=True, exist_ok=True)
        write_json(metric.with_suffix(".json"), {"dataset_sha256": "dataset", "reference_sha256": checksum})
        pd.DataFrame({"gop_idx": [0]}).to_parquet(metric.with_suffix(".gops.parquet"))
        pd.DataFrame({"frame_idx": range(30)}).to_parquet(metric.with_suffix(".frames.parquet"))
        cache = reports / "cache/v" / f"action_{action:02d}.parquet"
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame_idx": [0, 14, 29]}).to_parquet(cache)
        classes = cache.with_name(cache.stem + ".classes.parquet")
        pd.DataFrame({"class_id": [1]}).to_parquet(classes)
        write_json(cache.with_suffix(".json"), {"output_sha256": sha256(cache),
                   "classes_sha256": sha256(classes), "sample_frames": 3})
    last = reports / "cache/v/action_31.parquet"
    valid = last.read_bytes()
    last.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="benchmark cache"):
        compact.release_reference(ref, artifacts, reports, "dataset")
    assert reference_path.exists()
    last.write_bytes(valid)
    vmaf_log = artifacts / "metrics/v/action_00.vmaf.json"
    vmaf_log.write_bytes(b"disposable VMAF log")
    assert compact.release_reference(ref, artifacts, reports, "dataset") == len(b"disposable VMAF log")
    assert reference_path.read_bytes() == b"lossless original"
    assert not vmaf_log.exists()
    assert (artifacts / "reference/v.released.json").exists()
    assert compact.release_reference(ref, artifacts, reports, "dataset") == 0


def test_benchmark_reuses_verified_cache_without_reference(tmp_path, monkeypatch):
    from vcu_surrogate import assets
    monkeypatch.setattr(assets, "assert_downloads", lambda config: None)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "project_path", lambda config, group: tmp_path / "artifacts")
    monkeypatch.setattr(benchmark, "stable_hash", lambda value: "job")
    def fail_load(config):
        pytest.fail("fully cached benchmark should not load the detector")
    monkeypatch.setattr(benchmark, "_load_model", fail_load)
    artifacts = tmp_path / "artifacts"
    for name in ["reference", "encodes", "metadata"]:
        (artifacts / name).mkdir(parents=True)
    pd.DataFrame([{"video_id": "v", "reference_path": "missing.mkv",
                   "reference_sha256": "original", "frame_count": 30, "gop_count": 1}]).to_parquet(
        artifacts / "reference/manifest.parquet")
    encoded = artifacts / "encodes/action.mp4"
    encoded.write_bytes(b"encoded")
    pd.DataFrame([{"video_id": "v", "action_id": 0, "encoded_path": "artifacts/encodes/action.mp4",
                   "encoded_sha256": sha256(encoded)}]).to_parquet(artifacts / "encodes/manifest.parquet")
    pd.DataFrame(columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2", "confidence", "class_id"]).to_parquet(
        artifacts / "metadata/detections.parquet")
    output = tmp_path / "benchmark"
    cache = output / "cache/v/action_00.parquet"
    cache.parent.mkdir(parents=True)
    expected = pd.DataFrame({"frame_idx": [0, 14, 29]})
    expected.to_parquet(cache)
    classes = cache.with_name(cache.stem + ".classes.parquet")
    pd.DataFrame({"class_id": [1]}).to_parquet(classes)
    write_json(cache.with_suffix(".json"), {"job_hash": "job", "output_sha256": sha256(cache),
                                           "classes_sha256": sha256(classes)})
    samples, _ = benchmark.measure_grid({"_config_sha256": "config"}, output, 3, 8)
    pd.testing.assert_frame_equal(samples, expected)


def test_shared_reference_cache_preserves_metrics_and_rejects_invalid_inputs(tmp_path):
    import cv2
    from vcu_surrogate.common import run
    from vcu_surrogate.metrics import build_reference_yuv, build_reference_bgr, _run_vmaf, _roi_metrics
    reference = tmp_path / "ref.mkv"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
         "-frames:v", "3", "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-y", reference])
    checksum = sha256(reference)
    raw, bgr = tmp_path / "ref.yuv", tmp_path / "ref.npy"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_reference_yuv(reference, "wrong", 3, raw)
    assert not raw.exists()
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        build_reference_yuv(reference, checksum, 4, raw)
    assert not raw.exists()
    build_reference_yuv(reference, checksum, 3, raw)
    build_reference_bgr(reference, checksum, 3, bgr)
    assert raw.stat().st_size == 3 * 1920 * 1080 * 3 // 2
    cached = np.load(bgr, mmap_mode="r")
    cap = cv2.VideoCapture(str(reference), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    for frame in cached:
        ok, decoded = cap.read()
        assert ok
        np.testing.assert_array_equal(frame, decoded)
    cap.release()
    cached._mmap.close()
    boxes = pd.DataFrame([dict(frame_idx=i, x1=20, y1=30, x2=80, y2=90,
                              class_weight=1, confidence=.8) for i in range(3)])
    pd.testing.assert_frame_equal(_roi_metrics(reference, reference, boxes, 3),
                                  _roi_metrics(reference, reference, boxes, 3, bgr), check_exact=True)
    pd.testing.assert_frame_equal(_run_vmaf(reference, reference, tmp_path / "stream.json"),
                                  _run_vmaf(reference, reference, tmp_path / "raw.json", raw), check_exact=True)
    with pytest.raises(RuntimeError, match="extra frames"):
        build_reference_bgr(reference, checksum, 2, bgr)
    assert not bgr.exists()


def test_single_decode_stream_preserves_every_metric_and_cleans_failure(tmp_path, monkeypatch):
    import cv2
    from vcu_surrogate import metrics
    from vcu_surrogate.common import run
    reference = tmp_path / "reference.mkv"
    encoded = tmp_path / "encoded.mp4"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
         "-frames:v", "6", "-threads", "2", "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-y", reference])
    run(["ffmpeg", "-v", "error", "-i", reference, "-c:v", "libx264", "-threads", "2",
         "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p", "-y", encoded])
    raw = metrics.build_reference_yuv(reference, sha256(reference), 6, tmp_path / "reference.yuv")
    bgr = metrics.build_reference_bgr(reference, sha256(reference), 6, tmp_path / "reference.npy", raw)
    cached = np.load(bgr, mmap_mode="r")
    cap = cv2.VideoCapture(str(reference), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    for frame in cached:
        ok, decoded = cap.read()
        assert ok
        np.testing.assert_array_equal(frame, decoded)
    cap.release()
    cached._mmap.close()
    boxes = pd.DataFrame([dict(frame_idx=i, x1=20, y1=30, x2=100, y2=90,
                              class_weight=1., confidence=.8) for i in range(6)])
    expected_full = metrics._run_vmaf(encoded, reference, tmp_path / "baseline.json", raw)
    expected_roi = metrics._roi_metrics(encoded, reference, boxes, 6, bgr)
    children = []
    popen = metrics.subprocess.Popen
    def record(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append((args[0], child))
        return child
    monkeypatch.setattr(metrics.subprocess, "Popen", record)
    full, roi = metrics._stream_pair_metrics(encoded, reference, tmp_path / "stream.json", boxes, 6, raw, bgr)
    pd.testing.assert_frame_equal(full, expected_full, check_exact=True)
    pd.testing.assert_frame_equal(roi, expected_roi, check_exact=True)
    assert len(children) == 2
    assert sum(str(encoded) in command for command, child in children) == 1
    assert all(str(reference) not in command and child.poll() == 0 for command, child in children)
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"invalid bitstream")
    broken_log = tmp_path / "broken.json"
    with pytest.raises(RuntimeError, match="decode ended"):
        metrics._stream_pair_metrics(invalid, reference, broken_log, boxes, 6, raw, bgr)
    assert not broken_log.exists()
    assert all(child.poll() is not None for command, child in children)


def test_sampled_cache_migration_preserves_packets_and_resume(tmp_path, monkeypatch):
    from vcu_surrogate import metrics
    from vcu_surrogate.common import stable_hash
    monkeypatch.setattr(metrics, "ROOT", tmp_path)
    encoded = tmp_path / 'encoded.mp4'
    encoded.write_bytes(b'verified unchanged encoded stream')
    row = SimpleNamespace(video_id='v', action_id=0, encoded_path='encoded.mp4',
                          encoded_sha256=sha256(encoded), frame_count=60, bitrate_index=0,
                          roi_index=0, target_bitrate_mbps=1., delta_qp_roi=0)
    ref = SimpleNamespace(dataset_sha256='dataset', reference_sha256='reference')
    root = tmp_path / 'metrics'
    folder = root / 'v'
    folder.mkdir(parents=True)
    frames = pd.DataFrame({'frame_idx': range(60), 'gop_idx': np.arange(60)//30,
                           'vmaf': np.arange(60,dtype=float), 'psnr': 35., 'ssim': .9,
                           'roi_psnr': 30., 'roi_ssim': .8, 'roi_pixel_fraction': .1,
                           'has_roi': True, 'video_id': 'v', 'action_id': 0})
    frames.to_parquet(folder/'action_00.frames.parquet', index=False)
    packets = pd.DataFrame({'gop_idx':[0,1], 'payload_bytes':[1000,2000], 'key_packets':1,
                            'packet_count':30, 'actual_bitrate_mbps':[.008,.016],
                            'stream_container_overhead_bytes':10})
    packets.to_parquet(folder/'action_00.gops.parquet', index=False)
    job = metrics.metric_job(row, ref)
    write_json(folder/'action_00.json', {**job,'job_hash':stable_hash(job)})
    selected,gops = metrics.cached_metric(row,ref,root,30,30,5)
    assert selected.frame_idx.tolist() == list(range(0,60,5))
    assert gops.metric_sample_count.tolist() == [6,6]
    assert gops.frame_count.tolist() == [30,30]
    assert gops.vmaf.tolist() == [12.5,42.5]
    pd.testing.assert_frame_equal(gops[packets.columns], packets)
    a,b = metrics.cached_metric(row,ref,root,30,30,5)
    pd.testing.assert_frame_equal(a,selected)
    pd.testing.assert_frame_equal(b,gops)
    with pytest.raises(RuntimeError,match='incompatible metric'):
        metrics.cached_metric(row,ref,root,30,30,1)
    (folder/'action_00.frames.parquet').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        metrics.cached_metric(row,ref,root,30,30,5)


def test_subsample_preserves_temporal_context_and_cpu_cuda_parity(tmp_path):
    import os
    import json
    from vcu_surrogate import metrics
    from vcu_surrogate.common import run
    reference,encoded = tmp_path/'reference.mkv',tmp_path/'encoded.mp4'
    run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=1920x1080:rate=30',
         '-frames:v','30','-threads','2','-c:v','ffv1','-pix_fmt','yuv420p','-y',reference])
    run(['ffmpeg','-v','error','-i',reference,'-c:v','libx264','-threads','2',
         '-preset','veryfast','-crf','32','-pix_fmt','yuv420p','-y',encoded])
    raw = metrics.build_reference_yuv(reference,sha256(reference),30,tmp_path/'ref.yuv')
    bgr = metrics.build_reference_bgr(reference,sha256(reference),30,tmp_path/'ref.npy',raw)
    boxes = pd.DataFrame([dict(frame_idx=i,x1=20,y1=30,x2=100,y2=90,
                              class_weight=1.,confidence=.8) for i in range(30)])
    full = metrics._run_vmaf(encoded,reference,tmp_path/'full.json',raw)
    roi = metrics._roi_metrics(encoded,reference,boxes,30,bgr)
    metrics.build_reference_bgr(reference,sha256(reference),30,bgr,raw,5)
    assert np.load(bgr,mmap_mode='r').shape == (6,1080,1920,3)
    cpu,cpu_roi = metrics._stream_pair_metrics(encoded,reference,tmp_path/'cpu.json',boxes,30,raw,bgr,5,'cpu')
    assert cpu.frame_idx.tolist() == [0,5,10,15,20,25]
    pd.testing.assert_frame_equal(cpu, full.iloc[::5].reset_index(drop=True), check_exact=True)
    pd.testing.assert_frame_equal(cpu_roi, roi.iloc[::5].reset_index(drop=True), check_exact=True)
    if not Path('/opt/vmaf-cuda/bin/vmaf').exists():
        if os.environ.get('VCU_REQUIRE_CUDA_TEST') == '1':
            pytest.fail('required CUDA VMAF binary is missing')
        pytest.skip('CUDA VMAF runtime unavailable; CPU sampling verified')
    cuda,cuda_roi = metrics._stream_pair_metrics(encoded,reference,tmp_path/'cuda.json',boxes,30,raw,bgr,5,'cuda')
    np.testing.assert_array_equal(cpu.frame_idx,cuda.frame_idx)
    np.testing.assert_allclose(cpu.vmaf,cuda.vmaf,atol=1e-4,rtol=0)
    np.testing.assert_allclose(cpu[['psnr','ssim']],cuda[['psnr','ssim']],atol=1e-6,rtol=0)
    pd.testing.assert_frame_equal(cpu_roi,cuda_roi,check_exact=True)
    print('CUDA_PARITY',json.dumps({'vmaf_max_error':float(np.max(np.abs(cpu.vmaf-cuda.vmaf))),
                                  'psnr_ssim_max_error':float(np.max(np.abs(cpu[['psnr','ssim']].to_numpy()-cuda[['psnr','ssim']].to_numpy())))}))


def test_interrupted_metric_pair_commit_recovers_staged_tables(tmp_path, monkeypatch):
    from vcu_surrogate import metrics
    from vcu_surrogate.common import stable_hash
    row=SimpleNamespace(video_id='v',action_id=0,frame_count=30,
                        encoded_sha256='encoded',reference_sha256='reference')
    ref=SimpleNamespace(dataset_sha256='dataset',reference_sha256='reference')
    root=tmp_path/'metrics'; folder=root/'v'; folder.mkdir(parents=True)
    frame_path,gop_path,meta_path=(folder/f'action_00{s}' for s in ('.frames.parquet','.gops.parquet','.json'))
    old=pd.DataFrame({'frame_idx':range(30)})
    old.to_parquet(frame_path,index=False)
    pd.DataFrame({'gop_idx':[0]}).to_parquet(gop_path,index=False)
    frames=pd.DataFrame({'frame_idx':[0,5,10,15,20,25]})
    gops=pd.DataFrame({'gop_idx':[0],'metric_sample_count':[6],'frame_count':[30]})
    job=metrics.metric_job(row,ref,5)
    replace=Path.replace
    def interrupt(path,destination):
        if destination==gop_path:
            raise RuntimeError('simulated interruption')
        return replace(path,destination)
    monkeypatch.setattr(Path,'replace',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        metrics._save_metric_pair(frames,gops,frame_path,gop_path,meta_path,{**job,'job_hash':stable_hash(job)})
    monkeypatch.setattr(Path,'replace',replace)
    a,b=metrics.cached_metric(row,ref,root,30,30,5)
    pd.testing.assert_frame_equal(a,frames)
    pd.testing.assert_frame_equal(b,gops)
