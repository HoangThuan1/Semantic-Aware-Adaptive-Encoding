from __future__ import annotations

import json
import math
import ctypes
import subprocess
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ROOT, project_path, run, sha256, stable_hash, write_json


def _packet_metrics(path: Path, fps: int, gop_frames: int) -> pd.DataFrame:
    probe = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time,size,flags", "-of", "json", path,
    ], capture=True)
    packets = json.loads(probe.stdout)["packets"]
    rows = []
    for packet in packets:
        frame_idx = int(round(float(packet["pts_time"]) * fps))
        rows.append({"frame_idx": frame_idx, "packet_bytes": int(packet["size"]), "key_packet": "K" in packet["flags"]})
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"no video packets found in {path}")
    result["gop_idx"] = result.frame_idx // gop_frames
    return result


def build_reference_yuv(reference: Path, checksum: str, frame_count: int, output: Path) -> Path:
    """Decode the checked FFV1 once; share exact YUV420 planes across actions."""
    output.unlink(missing_ok=True)  # An interrupted raw cache has no trusted receipt.
    if sha256(reference) != checksum:
        raise RuntimeError("reference checksum mismatch before shared decode")
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,pix_fmt,avg_frame_rate", "-of", "json", reference], capture=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if (stream["width"], stream["height"], stream["pix_fmt"], stream["avg_frame_rate"]) != (1920, 1080, "yuv420p", "30/1"):
        raise RuntimeError("shared reference cache requires unchanged 1080p30 YUV420 planes")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", reference,
             "-map", "0:v:0", "-an", "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
             "-fps_mode", "passthrough", "-f", "rawvideo", "-y", output])
        if output.stat().st_size != frame_count * 1920 * 1080 * 3 // 2:
            raise RuntimeError("decoded reference cache frame coverage mismatch")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


class _Yuv420Bgr:
    """Use OpenCV's own swscale context/defaults, preserving its exact BGR pixels."""
    def __init__(self):
        import cv2
        candidates = list(Path(cv2.__file__).parent.parent.glob("opencv_python*.libs/libswscale*.so*"))
        loaded = Path("/proc/self/maps").read_text()
        candidates = [p for p in candidates if str(p) in loaded]
        if len(candidates) != 1:
            raise RuntimeError("cannot identify OpenCV's exact libswscale for shared decode")
        c = ctypes
        self.lib = c.CDLL(str(candidates[0]))
        self.pointer = c.POINTER(c.c_uint8)
        self.pointers, self.strides = self.pointer * 4, c.c_int * 4
        self.lib.sws_getContext.argtypes = [c.c_int] * 7 + [c.c_void_p, c.c_void_p, c.POINTER(c.c_double)]
        self.lib.sws_getContext.restype = c.c_void_p
        self.lib.sws_scale.argtypes = [c.c_void_p, self.pointers, self.strides,
                                      c.c_int, c.c_int, self.pointers, self.strides]
        self.lib.sws_scale.restype = c.c_int
        self.lib.sws_freeContext.argtypes = [c.c_void_p]
        self.context = self.lib.sws_getContext(1920, 1080, 0, 1920, 1080, 3, 4, None, None, None)
        if not self.context:
            raise RuntimeError("cannot create OpenCV-compatible YUV420/BGR context")
        self.output = np.empty((1080, 1920, 3), dtype=np.uint8)

    def convert(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8 or frame.size != 3110400 or not frame.flags.c_contiguous:
            raise RuntimeError("invalid YUV420 decoded frame")
        pointer = lambda addr: ctypes.cast(ctypes.c_void_p(addr), self.pointer)
        source = self.pointers(pointer(frame.ctypes.data), pointer(frame.ctypes.data + 2073600),
                               pointer(frame.ctypes.data + 2592000), None)
        destination = self.pointers(pointer(self.output.ctypes.data), None, None, None)
        rows = self.lib.sws_scale(self.context, source, self.strides(1920, 960, 960, 0),
                                  0, 1080, destination, self.strides(5760, 0, 0, 0))
        if rows != 1080:
            raise RuntimeError("incomplete YUV420/BGR conversion")
        return self.output

    def close(self):
        if self.context:
            self.lib.sws_freeContext(self.context)
            self.context = None


def build_reference_bgr(reference: Path, checksum: str, frame_count: int, output: Path,
                        reference_yuv: Path | None = None, sample_stride: int = 1) -> Path:
    """Build exact OpenCV BGR pixels, converting shared YUV when supplied."""
    import cv2
    output.unlink(missing_ok=True)
    if sha256(reference) != checksum:
        raise RuntimeError("reference checksum mismatch before shared BGR decode")
    if reference_yuv is not None:
        if reference_yuv.stat().st_size != frame_count * 3110400:
            raise RuntimeError("shared YUV reference coverage mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        converter = _Yuv420Bgr()
        raw = np.memmap(reference_yuv, mode="r", dtype=np.uint8, shape=(frame_count, 3110400))
        try:
            with output.open("wb") as handle:
                np.lib.format.write_array_header_1_0(handle, {
                    "descr": "|u1", "fortran_order": False, "shape": (len(range(0, frame_count, sample_stride)), 1080, 1920, 3)})
                for frame in raw[::sample_stride]:
                    handle.write(converter.convert(frame).data)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        finally:
            raw._mmap.close()
            converter.close()
        return output
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(reference), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = None
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot decode shared BGR reference")
        frames = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                          shape=(frame_count, 1080, 1920, 3))
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame.shape != (1080, 1920, 3):
                raise RuntimeError(f"shared BGR reference ended/mismatched at frame {index}")
            frames[index] = frame
        if cap.read()[0]:
            raise RuntimeError("shared BGR reference has extra frames")
        frames.flush()
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        if frames is not None:
            frames._mmap.close()
        cap.release()
    return output


def _vmaf_command(encoded: Path, reference: Path, log_path: Path,
                  reference_yuv: Path | None = None, raw_distorted: bool = False,
                  sample_stride: int = 1, backend: str = "cpu") -> list:
    if backend == "cuda":
        if reference_yuv is None or not raw_distorted:
            raise RuntimeError("CUDA VMAF requires shared raw reference and a streamed action")
        return ["/opt/vmaf-cuda/bin/vmaf", "--reference", str(reference_yuv),
                "--distorted", "/dev/stdin", "--width", "1920", "--height", "1080",
                "--pixel_format", "420", "--bitdepth", "8", "--model", "version=vmaf_v0.6.1",
                "--feature", "psnr", "--feature", "float_ssim", "--threads", "0",
                "--subsample", str(sample_stride), "--json", "--output", str(log_path), "--quiet"]
    if backend != "cpu":
        raise ValueError(f"unsupported metric backend: {backend}")
    graph = (
        "[0:v]settb=AVTB,setpts=N/(30*TB)[dist];"
        "[1:v]settb=AVTB,setpts=N/(30*TB)[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:"
        f"model=version=vmaf_v0.6.1:feature=name=psnr|name=float_ssim:n_threads=2:n_subsample={sample_stride}"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-filter_complex_threads", "1"]
    if raw_distorted:
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1920x1080",
                    "-framerate", "30", "-i", "pipe:0"]
    else:
        command += ["-threads", "2", "-i", encoded]
    if reference_yuv is None:
        command += ["-threads", "2", "-i", reference]
    else:
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1920x1080",
                    "-framerate", "30", "-i", reference_yuv]
    return command + ["-filter_complex", graph, "-f", "null", "-"]


def _run_vmaf(encoded: Path, reference: Path, log_path: Path,
              reference_yuv: Path | None = None) -> pd.DataFrame:
    log_path.unlink(missing_ok=True)
    run(_vmaf_command(encoded, reference, log_path, reference_yuv))
    return _read_vmaf(log_path)


def _read_vmaf(log_path: Path) -> pd.DataFrame:
    if not log_path.is_file():
        raise RuntimeError("libvmaf did not create its mandatory JSON log")
    data = json.loads(log_path.read_text())
    rows = []
    for frame in data["frames"]:
        metrics = frame["metrics"]
        try:
            rows.append({
                "frame_idx": int(frame["frameNum"]),
                "vmaf": float(metrics["vmaf"]),
                "psnr": float(metrics["psnr_y"]),
                "ssim": float(metrics["float_ssim"]),
            })
        except KeyError as error:
            raise RuntimeError(f"required libvmaf metric missing: {error}; available={sorted(metrics)}") from error
    if not rows:
        raise RuntimeError("libvmaf returned no per-frame metrics")
    return pd.DataFrame(rows)


def _roi_ssim(reference: np.ndarray, distorted: np.ndarray, win_size: int) -> float:
    """skimage's uniform-window RGB SSIM, using native float64 box filters.

    Same sample covariance, K1/K2, channel pooling and interior crop; only
    the filter implementation differs, within float64 rounding (1e-10).
    """
    import cv2

    if reference.shape != distorted.shape or reference.dtype != np.uint8 or distorted.dtype != np.uint8:
        raise ValueError("ROI SSIM requires matching uint8 RGB/BGR images")
    if reference.ndim != 3 or reference.shape[2] != 3 or win_size < 3 or win_size % 2 != 1:
        raise ValueError("ROI SSIM requires 3 channels and an odd window >= 3")
    if min(reference.shape[:2]) < win_size:
        raise ValueError("ROI SSIM window exceeds crop")
    pad = (win_size - 1) // 2
    cov_norm = win_size ** 2 / (win_size ** 2 - 1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = np.empty(3, dtype=np.float64)
    def box(image):
        return cv2.boxFilter(image, cv2.CV_64F, (win_size, win_size),
                             normalize=True, borderType=cv2.BORDER_REFLECT)
    for channel in range(3):
        x = reference[:, :, channel].astype(np.float64)
        y = distorted[:, :, channel].astype(np.float64)
        ux, uy = box(x), box(y)
        vx = cov_norm * (box(x * x) - ux * ux)
        vy = cov_norm * (box(y * y) - uy * uy)
        vxy = cov_norm * (box(x * y) - ux * uy)
        score = ((2 * ux * uy + c1) * (2 * vxy + c2)
                 / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)))
        scores[channel] = score[pad:-pad, pad:-pad].mean(dtype=np.float64)
    return float(scores.mean())


def _roi_metrics(encoded: Path, reference: Path, detections: pd.DataFrame, frame_count: int,
                 reference_bgr: Path | None = None, distorted_frames=None,
                 sample_stride: int = 1) -> pd.DataFrame:
    import cv2

    # Bound each worker's decoder buffers and avoid competing OpenCV thread pools.
    cv2.setNumThreads(1)
    decode_params = [cv2.CAP_PROP_N_THREADS, 2]
    encoded_cap = cv2.VideoCapture(str(encoded), cv2.CAP_FFMPEG, decode_params) if distorted_frames is None else None
    cached = np.load(reference_bgr, mmap_mode="r") if reference_bgr is not None else None
    if cached is not None and (cached.dtype != np.uint8 or cached.shape not in
            ((frame_count, 1080, 1920, 3), (len(range(0, frame_count, sample_stride)), 1080, 1920, 3))):
        raise RuntimeError("invalid decoded reference cache coverage/shape")
    reference_cap = cv2.VideoCapture(str(reference), cv2.CAP_FFMPEG, decode_params) if cached is None else None
    if ((encoded_cap is not None and not encoded_cap.isOpened()) or
            (reference_cap is not None and not reference_cap.isOpened())):
        raise RuntimeError(f"cannot decode ROI metric pair: {encoded}, {reference}")
    by_frame = {int(idx): group for idx, group in detections.groupby("frame_idx")}
    rows = []
    indices = range(0, frame_count, sample_stride) if distorted_frames is not None else range(frame_count)
    for frame_idx in tqdm(indices, desc=f"ROI metrics {encoded.stem}", leave=False, disable=True):
        if encoded_cap is not None:
            ok_dist, distorted = encoded_cap.read()
        else:
            distorted = next(distorted_frames, None)
            ok_dist = distorted is not None
        ok_ref, original = reference_cap.read() if reference_cap is not None else (True, cached[frame_idx if len(cached) == frame_count else frame_idx // sample_stride])
        if not ok_dist or not ok_ref:
            raise RuntimeError(f"metric decode ended at frame {frame_idx}: {encoded}")
        if frame_idx % sample_stride:
            continue
        group = by_frame.get(frame_idx)
        if group is None or group.empty:
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        height, width = original.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        ssim_weighted = 0.0
        weight_sum = 0.0
        for box in group.itertuples(index=False):
            x1, y1 = max(0, int(math.floor(box.x1))), max(0, int(math.floor(box.y1)))
            x2, y2 = min(width, int(math.ceil(box.x2))), min(height, int(math.ceil(box.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
            ref_crop = original[y1:y2, x1:x2]
            dist_crop = distorted[y1:y2, x1:x2]
            minimum = min(ref_crop.shape[0], ref_crop.shape[1])
            win_size = min(7, minimum if minimum % 2 == 1 else minimum - 1)
            if win_size >= 3:
                score = _roi_ssim(ref_crop, dist_crop, win_size)
                weight = float(box.class_weight * box.confidence * (x2 - x1) * (y2 - y1))
                ssim_weighted += score * weight
                weight_sum += weight
        if not mask.any():
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        difference = original[mask].astype(np.float32) - distorted[mask].astype(np.float32)
        mse = float(np.mean(np.square(difference)))
        roi_psnr = 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
        rows.append({
            "frame_idx": frame_idx, "has_roi": True,
            "roi_pixel_fraction": float(mask.mean()), "roi_psnr": roi_psnr,
            "roi_ssim": ssim_weighted / weight_sum if weight_sum else np.nan,
        })
    if encoded_cap is not None:
        encoded_cap.release()
    if reference_cap is not None:
        reference_cap.release()
    return pd.DataFrame(rows)


def _stream_pair_metrics(encoded: Path, reference: Path, log: Path, detections: pd.DataFrame,
                         frame_count: int, reference_yuv: Path, reference_bgr: Path,
                         sample_stride: int = 1, backend: str = "cpu"):
    """One action decode; fan each YUV frame to libvmaf and exact BGR ROI metrics."""
    log.unlink(missing_ok=True)
    converter = _Yuv420Bgr()
    decoder = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(encoded),
        "-map", "0:v:0", "-an", "-pix_fmt", "yuv420p", "-threads", "1",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    vmaf = None
    try:
        environment = dict(os.environ)
        if backend == "cuda":
            environment["LD_LIBRARY_PATH"] = "/opt/vmaf-cuda/lib:" + environment.get("LD_LIBRARY_PATH", "")
        vmaf = subprocess.Popen(_vmaf_command(encoded, reference, log, reference_yuv, True,
                                             sample_stride, backend),
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, env=environment)
        def frames():
            for index in range(frame_count):
                raw = decoder.stdout.read(3110400)
                if len(raw) != 3110400:
                    raise RuntimeError(f"single metric decode ended at frame {index}")
                vmaf.stdin.write(raw)
                if index % sample_stride == 0:
                    yield converter.convert(np.frombuffer(raw, dtype=np.uint8))
            if decoder.stdout.read(1):
                raise RuntimeError("single metric decode has extra frames")
        stream = frames()
        roi = _roi_metrics(encoded, reference, detections, frame_count, reference_bgr, stream, sample_stride)
        next(stream, None)  # Check decoder EOF after consuming exactly frame_count frames.
        vmaf.stdin.close()
        decode_code, vmaf_code = decoder.wait(), vmaf.wait()
        if decode_code or vmaf_code:
            raise RuntimeError(f"single metric decode/VMAF failed: {decode_code}/{vmaf_code}")
        full = _read_vmaf(log)
        if full.frame_idx.tolist() != list(range(0, frame_count, sample_stride)):
            raise RuntimeError("single decode VMAF frame coverage mismatch")
        return full, roi
    except BaseException:
        log.unlink(missing_ok=True)
        raise
    finally:
        for child in (decoder, vmaf):
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        decoder.stdout.close()
        if vmaf is not None:
            try:
                vmaf.stdin.close()
            except BrokenPipeError:
                pass
        converter.close()


def metric_job(encode, reference, sample_stride=1):
    job = {"dataset_sha256": reference.dataset_sha256,
           "encoded_sha256": encode.encoded_sha256, "reference_sha256": reference.reference_sha256,
           "metrics": ["vmaf", "psnr_y", "float_ssim", "roi_psnr", "roi_ssim"]}
    if sample_stride != 1:
        job["sampling"] = {"stride": sample_stride, "offset": 0, "temporal_context": "all_frames"}
    return job


def _aggregate_quality(frames, packet_gops, encode, gop_frames):
    quality = frames.groupby("gop_idx", as_index=False).agg(
        vmaf=("vmaf", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"),
        roi_psnr=("roi_psnr", "mean"), roi_ssim=("roi_ssim", "mean"),
        roi_pixel_fraction=("roi_pixel_fraction", "mean"), roi_frame_count=("has_roi", "sum"),
        metric_sample_count=("frame_idx", "size"))
    gops = quality.merge(packet_gops, on="gop_idx", validate="one_to_one")
    gops["frame_count"] = gop_frames
    for field in ("video_id", "action_id", "bitrate_index", "roi_index", "target_bitrate_mbps", "delta_qp_roi"):
        gops[field] = getattr(encode, field)
    return gops


def _save_metric_pair(frames, gops, frame_output, gop_output, sidecar, metadata):
    # Record both staged hashes before committing either table; every crash point is resumable.
    staged = []
    target = dict(metadata)
    for frame, path, field in ((frames, frame_output, "frames_sha256"),
                               (gops, gop_output, "gops_sha256")):
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        staged.append((temporary, path))
        target[field] = sha256(temporary)
    write_json(sidecar, {**target, "commit_pending": True})
    for temporary, path in staged:
        temporary.replace(path)
    write_json(sidecar, target)


def cached_metric(encode, reference, metrics_root, fps, gop_frames, sample_stride=1):
    prefix = metrics_root / encode.video_id / f"action_{encode.action_id:02d}"
    frame_output, gop_output, sidecar = (prefix.with_suffix(s) for s in
                                       (".frames.parquet", ".gops.parquet", ".json"))
    if not all(p.is_file() for p in (frame_output, gop_output, sidecar)):
        return None
    metadata = json.loads(sidecar.read_text())
    if metadata.get("commit_pending"):
        for path, field in ((frame_output, "frames_sha256"), (gop_output, "gops_sha256")):
            temporary = path.with_name(path.name + ".tmp")
            if sha256(path) == metadata[field]:
                temporary.unlink(missing_ok=True)
            elif temporary.is_file() and sha256(temporary) == metadata[field]:
                temporary.replace(path)
            else:
                raise RuntimeError(f"cannot resume metric commit: {path}")
        metadata.pop("commit_pending")
        write_json(sidecar, metadata)
    job = metric_job(encode, reference, sample_stride)
    expected, legacy = stable_hash(job), stable_hash(metric_job(encode, reference, 1))
    if metadata.get("job_hash") not in {expected, legacy}:
        raise RuntimeError(f"incompatible metric artifact: {gop_output}")
    for field, path in (("frames_sha256", frame_output), ("gops_sha256", gop_output)):
        if field in metadata and metadata[field] != sha256(path):
            raise RuntimeError(f"metric checksum mismatch: {path}")
    frames, gops = pd.read_parquet(frame_output), pd.read_parquet(gop_output)
    indices = list(range(0, int(encode.frame_count), sample_stride))
    if metadata["job_hash"] == expected:
        if frames.frame_idx.tolist() != indices:
            raise RuntimeError("cached metric frame coverage mismatch")
        if sample_stride != 1 and (not (gops.metric_sample_count == gop_frames // sample_stride).all()
                                   or not (gops.frame_count == gop_frames).all()):
            raise RuntimeError("cached metric GOP sample coverage mismatch")
        return frames, gops
    if frames.frame_idx.tolist() not in (list(range(int(encode.frame_count))), indices):
        raise RuntimeError("cannot sample incomplete legacy metric frames")
    if gops.gop_idx.tolist() != list(range(int(encode.frame_count) // gop_frames)):
        raise RuntimeError("cannot sample incomplete legacy metric GOPs")
    if sha256(ROOT / encode.encoded_path) != encode.encoded_sha256:
        raise RuntimeError("encode checksum mismatch before metric conversion")
    frames = frames[frames.frame_idx % sample_stride == 0].copy().reset_index(drop=True)
    packet_fields = ["gop_idx", "payload_bytes", "key_packets", "packet_count", "actual_bitrate_mbps",
                     "stream_container_overhead_bytes"]
    gops = _aggregate_quality(frames, gops[packet_fields], encode, gop_frames)
    _save_metric_pair(frames, gops, frame_output, gop_output, sidecar,
                      {**metadata, **job, "job_hash": expected,
                       "sampled_from_full_job_hash": legacy, "metric_sample_stride": sample_stride,
                       "metric_sample_offset": 0, "backend": metadata.get("backend", "cpu"),
                       "conversion": "selected existing exact frame results; no decode"})
    return frames, gops


def _metric_one(encode, reference: pd.Series, detections: pd.DataFrame,
                metrics_root: Path, fps: int, gop_frames: int,
                reference_yuv: Path | None = None, reference_bgr: Path | None = None,
                sample_stride: int = 1, backend: str = "cpu") -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_stride < 1 or gop_frames % sample_stride:
        raise ValueError("metric stride must divide GOP size")
    existing = cached_metric(encode, reference, metrics_root, fps, gop_frames, sample_stride)
    if existing is not None:
        return existing
    encoded_path, reference_path = ROOT / encode.encoded_path, ROOT / reference.reference_path
    video_dir = metrics_root / encode.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    prefix = video_dir / f"action_{encode.action_id:02d}"
    frame_output, gop_output, sidecar = (prefix.with_suffix(s) for s in
                                       (".frames.parquet", ".gops.parquet", ".json"))
    job = metric_job(encode, reference, sample_stride)
    vmaf_log = prefix.with_suffix(".vmaf.json")
    started = time.monotonic()
    if reference_yuv is not None and reference_bgr is not None:
        full, roi = _stream_pair_metrics(encoded_path, reference_path, vmaf_log, detections,
                                         int(encode.frame_count), reference_yuv, reference_bgr,
                                         sample_stride, backend)
    else:
        if backend != "cpu":
            raise RuntimeError("CUDA metrics require shared reference caches")
        run(_vmaf_command(encoded_path, reference_path, vmaf_log, reference_yuv,
                          sample_stride=sample_stride))
        full = _read_vmaf(vmaf_log)
        roi = _roi_metrics(encoded_path, reference_path, detections, int(encode.frame_count),
                           reference_bgr, sample_stride=sample_stride)
    frames = full.merge(roi, on="frame_idx", validate="one_to_one")
    if frames.frame_idx.tolist() != list(range(0, int(encode.frame_count), sample_stride)):
        raise RuntimeError(f"metric frame count mismatch: {encoded_path}")
    frames["video_id"], frames["action_id"] = encode.video_id, encode.action_id
    frames["gop_idx"] = frames.frame_idx // gop_frames
    packets = _packet_metrics(encoded_path, fps, gop_frames)
    packet_gops = packets.groupby("gop_idx", as_index=False).agg(
        payload_bytes=("packet_bytes", "sum"), key_packets=("key_packet", "sum"),
        packet_count=("packet_bytes", "size"))
    packet_gops["actual_bitrate_mbps"] = packet_gops.payload_bytes * 8.0 / (gop_frames / fps) / 1_000_000
    packet_gops["stream_container_overhead_bytes"] = encoded_path.stat().st_size - int(packets.packet_bytes.sum())
    gops = _aggregate_quality(frames, packet_gops, encode, gop_frames)
    _save_metric_pair(frames, gops, frame_output, gop_output, sidecar,
                      {**job, "job_hash": stable_hash(job), "backend": backend,
                       "vmaf_threads": 2 if backend == "cpu" else 0, "metric_workers": 6,
                       "vmaf_cuda_commit": "17a67b238ce0539bdeafdc95961abac64fa16ea8" if backend == "cuda" else None,
                       "metric_sample_stride": sample_stride, "metric_sample_offset": 0,
                       "elapsed_s": time.monotonic() - started,
                       "action_decode_passes": 1 if reference_yuv is not None and reference_bgr is not None else 2,
                       "bgr_conversion": "opencv_bundled_swscale", "roi_ssim_backend": "opencv_float64_boxfilter",
                       "roi_ssim_equivalence_atol": 1e-10})
    return frames, gops


_WORKER_BACKEND = "cpu"


def _init_metric_backend(tokens):
    global _WORKER_BACKEND
    _WORKER_BACKEND = tokens.get()


def _mixed_metric_one(args):
    return _metric_one(*args, backend=_WORKER_BACKEND)


def mixed_metric_results(jobs, cpu_workers=5, cuda_workers=1):
    """One common action queue; backend is fixed per process, never per action ID."""
    if not jobs:
        return
    context = mp.get_context("spawn")
    tokens = context.Queue()
    for backend in ["cuda"] * cuda_workers + ["cpu"] * cpu_workers:
        tokens.put(backend)
    try:
        with ProcessPoolExecutor(max_workers=cpu_workers + cuda_workers, mp_context=context,
                                 initializer=_init_metric_backend, initargs=(tokens,)) as pool:
            futures = {pool.submit(_mixed_metric_one, args): args[0].action_id for args in jobs}
            try:
                for future in as_completed(futures):
                    action = futures.pop(future)
                    yield action, future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        tokens.close()
        tokens.join_thread()


def compute_metrics(config: dict) -> None:
    artifacts = project_path(config, "artifacts")
    encode_manifest_path = artifacts / "encodes" / "manifest.parquet"
    reference_manifest_path = artifacts / "reference" / "manifest.parquet"
    detections_path = artifacts / "metadata" / "detections.parquet"
    if not all(path.is_file() for path in (encode_manifest_path, reference_manifest_path, detections_path)):
        raise RuntimeError("encode manifest, reference manifest, or detections missing")
    encodes = pd.read_parquet(encode_manifest_path)
    references = pd.read_parquet(reference_manifest_path).set_index("video_id")
    detections = pd.read_parquet(detections_path)
    detections_by_video = {video_id: frame for video_id, frame in detections.groupby("video_id")}
    fps = config["video"]["fps"]
    gop_frames = config["video"]["gop_frames"]
    stride = config.get("metrics", {}).get("sample_stride", 1)
    metrics_root = artifacts / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    from .parquet_stream import ParquetStream
    grid_parts = []
    with ParquetStream(metrics_root / "frame_metrics.parquet") as writer:
        with ProcessPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_metric_one, SimpleNamespace(**encode._asdict()),
                                   references.loc[encode.video_id],
                                   detections_by_video.get(encode.video_id, detections.iloc[0:0]),
                                   metrics_root, fps, gop_frames, None, None, stride)
                       for encode in encodes.itertuples(index=False)}
            for future in as_completed(futures):
                frames, gops = future.result()
                writer.append(frames)
                grid_parts.append(gops)
                futures.remove(future)
    pd.concat(grid_parts, ignore_index=True).sort_values(
        ["video_id", "gop_idx", "action_id"]
    ).to_parquet(metrics_root / "gop_action_grid.parquet", index=False)
