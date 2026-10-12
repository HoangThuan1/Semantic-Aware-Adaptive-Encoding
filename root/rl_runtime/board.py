"""ZCU106 Vivado-flow runner. No VVAS, CPU detector or software encoder path."""
import argparse
import io
import json
import os
import resource
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import (FPS, FRAMES, FRAME_BYTES, GOP, HEIGHT, WIDTH, CLASS_WEIGHTS,
                     STRATEGIES, bundle_manifest, decision, memory_budget, read_trace,
                     roi_boxes, semantic, sha256, validate_au, write_json)
from .policy_numpy import NumpyPolicy


BOARD_PREPROCESSING = {
    "scaler": "v4l2convert", "scaler_rgb": [1280, 720],
    "arm_resize_rgb": [640, 360], "arm_resize": "OpenCV INTER_LINEAR",
    "letterbox_rgb": [640, 640], "pad_top_bottom": 140, "pad_value": 114,
}


def resize_rgb_for_dpu(rgb):
    """Mandatory board preprocessing, not a CPU inference fallback."""
    if rgb.shape != (720, 1280, 3) or rgb.dtype != np.uint8:
        raise RuntimeError("requires scaler RGB uint8 1280x720")
    import cv2
    return cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)


def hardware_imports():
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo
    import xir
    import vart
    Gst.init(None)
    return Gst, GstVideo, xir, vart


def available_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def global_nms(core, rows, threshold, limit):
    """Global confidence order, matching YOLOv5 class-aware NMS/max_det."""
    rows = rows[np.argsort(-rows[:, 4], kind="stable")]
    kept = []
    while len(rows) and len(kept) < limit:
        first = rows[0]
        kept.append(first)
        rest = rows[1:]
        overlap = core.box_iou_one_to_many(first[:4], rest[:, :4])
        rows = rest[(rest[:, 5] != first[5]) | (overlap <= threshold)]
    return np.asarray(kept, dtype=np.float32).reshape(-1, 6)


class Dpu:
    def __init__(self, root, manifest, timeout):
        from . import dpu_core as core
        import cv2
        cv2.setNumThreads(1)
        if manifest["detector"].get("board_preprocessing") != BOARD_PREPROCESSING:
            raise RuntimeError("manifest/board preprocessing contract mismatch")
        _, _, xir, vart = hardware_imports()
        self.core, self.timeout = core, timeout
        self.graph = xir.Graph.deserialize(str(root / manifest["detector"]["xmodel"]))
        subgraphs = core.get_dpu_subgraphs(self.graph)
        if len(subgraphs) != 1:
            raise RuntimeError("requires exactly one DPU subgraph; unsupported split graph")
        self.subgraph = subgraphs[0]
        self.runner = vart.Runner.create_runner(self.subgraph, "run")
        inputs = self.runner.get_input_tensors()
        self.outputs_tensors = self.runner.get_output_tensors()
        if len(inputs) != 1 or len(self.outputs_tensors) != 3:
            raise RuntimeError("requires one input and three raw YOLOv5n heads")
        self.input_tensor = inputs[0]
        self.shape = core.tensor_dims(self.input_tensor)
        self.layout, h, w, channels = core.infer_input_layout(self.shape)
        if self.shape[0] != 1 or (h, w, channels) != (640, 640, 3):
            raise RuntimeError("XModel does not meet pinned batch1/640x640/RGB contract")
        for tensor in [self.input_tensor] + list(self.outputs_tensors):
            if core.tensor_fix_point(tensor) is None or "8" not in str(tensor.dtype) or "16" in str(tensor.dtype):
                raise RuntimeError("requires explicit INT8/fix_point tensors: " + tensor.name)
        fix = core.tensor_fix_point(self.input_tensor)
        self.lut = np.clip(np.round(np.arange(256, dtype=np.float32)/255 * 2**fix), -128, 127).astype(np.int8)
        self.outputs = [np.empty(core.tensor_dims(t), dtype=np.int8) for t in self.outputs_tensors]
        self.anchors = np.asarray(json.loads((root / "models/yolov5n/anchors.json").read_text()), dtype=np.float32)
        if self.anchors.shape != (3, 3, 2) or not np.isfinite(self.anchors).all() or (self.anchors <= 0).any():
            raise RuntimeError("invalid explicit anchors")
        self.tensor_report = {"input": {"name": self.input_tensor.name, "shape": self.shape, "fix_point": fix},
                              "outputs": [{"name": t.name, "shape": core.tensor_dims(t),
                                           "fix_point": core.tensor_fix_point(t)} for t in self.outputs_tensors],
                              "subgraph": self.subgraph.get_name()}
        if not self.subgraph.has_attr("dpu_fingerprint"):
            raise RuntimeError("XModel has no dpu_fingerprint")
        expected = int(self.subgraph.get_attr("dpu_fingerprint"))
        query = subprocess.run(["xdputil", "query"], check=True, capture_output=True, text=True, timeout=timeout)
        # xdputil releases differ in the outer JSON shape; fingerprint keys are explicit.
        start = query.stdout.find("{")
        report = json.loads(query.stdout[start:])
        found = []
        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if "fingerprint" in key.lower():
                        found.append(int(str(value), 0) if isinstance(value, str) else int(value))
                    else:
                        visit(value)
            elif isinstance(obj, list):
                for value in obj:
                    visit(value)
        visit(report)
        if not found or any(value != expected for value in found):
            raise RuntimeError("hardware/XModel DPU fingerprint mismatch")
        self.tensor_report.update({"fingerprint": hex(expected), "xdputil": report,
                                   "board_preprocessing": BOARD_PREPROCESSING, "opencv": cv2.__version__})

    def infer(self, rgb):
        start = time.perf_counter()
        rgb = resize_rgb_for_dpu(rgb)
        resized = time.perf_counter()
        image = np.full((640, 640, 3), self.lut[114], dtype=np.int8)
        image[140:500] = self.lut[rgb]
        if self.layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        inp = np.ascontiguousarray(image.reshape(self.shape))
        prepared = time.perf_counter()
        job = self.runner.execute_async([inp], self.outputs)
        status = self.runner.wait(job, int(self.timeout*1000))
        if status != 0:
            raise RuntimeError("DPU timeout/error: " + str(status))
        inferred = time.perf_counter()
        parts, strides = [], []
        for raw, tensor in zip(self.outputs, self.outputs_tensors):
            fp = self.core.dequantize_output(raw, tensor)
            head = self.core.normalize_yolo_head(fp, 80)
            index, stride, anchors = self.core.match_anchors_to_head(head, self.anchors, 640, 640)
            if stride not in (8, 16, 32) or index in strides:
                raise RuntimeError("unsupported/duplicate YOLO head stride")
            strides.append(index)
            rows = self.core.decode_head(head, anchors, 640, 640, .25)
            rows = rows[np.isin(rows[:, 5].astype(int), list(CLASS_WEIGHTS))]
            parts.append(rows)
        rows = global_nms(self.core, np.concatenate(parts), .45, 32)
        rows[:, [0, 2]] *= 3
        rows[:, [1, 3]] = (rows[:, [1, 3]]-140)*3
        rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, WIDTH)
        rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, HEIGHT)
        ended = time.perf_counter()
        return rows, {"arm_resize_ms": (resized-start)*1000,
                      "quantize_ms": (prepared-resized)*1000, "dpu_ms": (inferred-prepared)*1000,
                      "postprocess_ms": (ended-inferred)*1000}


class Endpoint:
    def __init__(self, text, timeout):
        self.Gst, self.GstVideo, _, _ = hardware_imports()
        self.timeout, self.text = timeout, text
        self.pipeline = self.Gst.parse_launch(text)
        self.src, self.sink = self.pipeline.get_by_name("src"), self.pipeline.get_by_name("sink")
        if not self.src or not self.sink:
            raise RuntimeError("missing app endpoints")

    def start(self):
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer PLAYING failed")

    def check_bus(self):
        msg = self.pipeline.get_bus().pop_filtered(self.Gst.MessageType.ERROR)
        if msg:
            error, debug = msg.parse_error()
            raise RuntimeError(str(error) + ": " + str(debug))

    def push(self, payload, frame, rois=(), delta=0):
        self.check_bus()
        buffer = self.Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, bytes(payload))
        buffer.pts = frame*1_000_000_000//FPS
        buffer.dts = self.Gst.CLOCK_TIME_NONE
        buffer.duration = (frame+1)*1_000_000_000//FPS-buffer.pts
        buffer.offset = frame
        for roi in rois:
            meta = self.GstVideo.buffer_add_video_region_of_interest_meta(
                buffer, str(roi["class_id"]), roi["x"], roi["y"], roi["width"], roi["height"])
            if meta is None:
                raise RuntimeError("cannot attach ROI metadata")
            param = self.Gst.Structure.new_empty("roi-by-value/omx-alg")
            param.set_value("delta-qp", int(delta))
            meta.add_param(param)
            if meta.get_param("roi-by-value/omx-alg") is None:
                raise RuntimeError("ROI param did not survive attachment")
        if self.src.emit("push-buffer", buffer) != self.Gst.FlowReturn.OK:
            raise RuntimeError("appsrc push failure")

    def pull(self):
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            self.check_bus()
            sample = self.sink.emit("try-pull-sample", 100_000_000)
            if sample is not None:
                return sample
            if self.sink.get_property("eos"):
                raise RuntimeError("early appsink EOS")
        raise RuntimeError("GStreamer appsink timeout")

    def rgb(self, sample, frame):
        buf = sample.get_buffer()
        if abs(int(buf.pts)-frame*1_000_000_000//FPS) > 1:
            raise RuntimeError("scaler changed/reordered PTS")
        info = self.GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        if (info.width, info.height) != (1280, 720) or info.finfo.name != "RGB":
            raise RuntimeError("scaler caps mismatch")
        meta = self.GstVideo.buffer_get_video_meta(buf)
        stride = int(meta.stride[0] if meta else info.stride[0])
        offset = int(meta.offset[0] if meta else info.offset[0])
        ok, mapped = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("cannot map scaler buffer")
        try:
            if stride < 1280*3 or offset+(720-1)*stride+1280*3 > len(mapped.data):
                raise RuntimeError("invalid negotiated RGB stride/offset")
            return np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=mapped.data,
                              offset=offset, strides=(stride, 3, 1)).copy()
        finally:
            buf.unmap(mapped)

    def nv12(self, sample, frame):
        buf = sample.get_buffer()
        if abs(int(buf.pts)-frame*1_000_000_000//FPS) > 1:
            raise RuntimeError("ingress changed/reordered PTS")
        info = self.GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        if (info.width, info.height) != (WIDTH, HEIGHT) or info.finfo.name != "NV12":
            raise RuntimeError("ingress NV12 caps mismatch")
        meta = self.GstVideo.buffer_get_video_meta(buf)
        stride = meta.stride if meta else info.stride
        offset = meta.offset if meta else info.offset
        ok, mapped = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("cannot map ingress NV12 buffer")
        try:
            planes = []
            for plane, height in ((0, HEIGHT), (1, HEIGHT//2)):
                if stride[plane] < WIDTH or offset[plane]+(height-1)*stride[plane]+WIDTH > len(mapped.data):
                    raise RuntimeError("invalid NV12 negotiated stride/offset")
                pixels = np.ndarray((height, WIDTH), dtype=np.uint8, buffer=mapped.data,
                                    offset=int(offset[plane]), strides=(int(stride[plane]), 1))
                planes.append(pixels.tobytes())
            return b"".join(planes)
        finally:
            buf.unmap(mapped)

    def eos(self):
        if self.src.emit("end-of-stream") != self.Gst.FlowReturn.OK:
            raise RuntimeError("appsrc EOS failure")
        msg = self.pipeline.get_bus().timed_pop_filtered(int(self.timeout*1e9),
                    self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if msg is None:
            raise RuntimeError("GStreamer EOS timeout")
        if msg.type == self.Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            raise RuntimeError(str(err)+": "+str(debug))
        if self.sink.emit("try-pull-sample", 0) is not None:
            raise RuntimeError("unexpected extra frame at EOS")

    def close(self):
        self.pipeline.set_state(self.Gst.State.NULL)


def pipeline_texts(strategy, device):
    if any(c in device for c in ('"', "\\", "\n")):
        raise ValueError("invalid V4L2 device path")
    caps = ('video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1,'
            'interlace-mode=progressive,colorimetry=bt709,chroma-site=mpeg2')
    source = ('appsrc name=src is-live=false format=time block=false do-timestamp=false '
              'max-bytes=%d caps="%s" ! ' % (GOP*FRAME_BYTES, caps))
    scaler = (source + 'queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! '
              'v4l2convert capture-io-mode=4 output-io-mode=4 device="%s" ! ' % device +
              'video/x-raw,format=RGB,width=1280,height=720,colorimetry=sRGB ! '
              'appsink name=sink sync=false max-buffers=1 drop=false')
    roi = strategy in ("fixed_roi", "rl")
    encoder = (source + 'queue max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
               'omxh264enc name=vcu gop-mode=basic gop-length=30 periodicity-idr=30 b-frames=0 '
               'target-bitrate=1500 control-rate=%s qp-mode=%s filler-data=false ! ' %
               ("constant" if strategy == "cbr" else "variable", "roi" if roi else "uniform") +
               'video/x-h264,alignment=au ! h264parse config-interval=-1 ! '
               'video/x-h264,alignment=au,stream-format=byte-stream ! '
               'appsink name=sink sync=false max-buffers=30 drop=false')
    return scaler, encoder


def ingress_text():
    return ('appsrc name=src is-live=false format=time block=false do-timestamp=false '
            'max-bytes=%d caps="video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1,colorimetry=bt709" ! '
            'queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! '
            'appsink name=sink sync=false max-buffers=1 drop=false' % FRAME_BYTES)


def preflight(root, args):
    started = time.perf_counter()
    manifest = bundle_manifest(root)
    read_trace(root, manifest)
    policy = NumpyPolicy(root / "models/rl/ddqn_policy.npz")
    golden = json.loads((root / "models/rl/golden_policy.json").read_text())
    if [row["gop_idx"] for row in golden] != list(range(4, FRAMES//GOP)):
        raise RuntimeError("independent golden policy states are missing/incomplete")
    q_errors = []
    for row in golden:
        q = policy.q_values(row["raw_state"])[0]
        if not np.allclose(q, row["q_values"], rtol=1e-5, atol=1e-5) or int(q.argmax()) != row["action_id"]:
            raise RuntimeError("ARM policy replay differs from golden PyTorch")
        q_errors.append(float(np.max(np.abs(q-np.asarray(row["q_values"])))))
    swaps = Path("/proc/swaps").read_text().splitlines()
    if len(swaps) > 1:
        raise RuntimeError("swap is enabled; DDR benchmark requires swap disabled before launch")
    Gst, GstVideo, _, _ = hardware_imports()
    for name in ("appsrc", "appsink", "queue", "v4l2convert", "omxh264enc", "h264parse"):
        if Gst.ElementFactory.find(name) is None:
            raise RuntimeError("missing required element: " + name)
    if not Path(args.device).is_char_device():
        raise RuntimeError("Multi-Scaler V4L2 device missing: " + args.device)
    inspector = subprocess.run(["gst-inspect-1.0", "omxh264enc"], check=True, capture_output=True,
                               text=True, timeout=args.timeout).stdout
    import re
    if not re.search(r"target-bitrate[^\n]*(?:kbit|kbps|kbits|kilobit)", inspector, re.I):
        raise RuntimeError("cannot verify omxh264enc target-bitrate units as kbit/s")
    instance = Gst.ElementFactory.make("omxh264enc", "test")
    prop = instance.find_property("target-bitrate")
    if prop is None or not (int(prop.flags) & 4096):
        raise RuntimeError("target-bitrate is not mutable in PLAYING")
    pipelines = {}
    for strategy in STRATEGIES:
        texts = pipeline_texts(strategy, args.device)
        endpoints = []
        try:
            for text in texts:
                endpoints.append(Endpoint(text, args.timeout))
            encoder = endpoints[1].pipeline.get_by_name("vcu")
            for name, expected in (("gop-length", 30), ("periodicity-idr", 30), ("b-frames", 0)):
                if encoder.get_property(name) != expected:
                    raise RuntimeError("encoder property mismatch: " + name)
            if strategy in ("fixed_roi", "rl"):
                b = Gst.Buffer.new_allocate(None, FRAME_BYTES, None)
                meta = GstVideo.buffer_add_video_region_of_interest_meta(b, "test", 160, 160, 320, 320)
                param = Gst.Structure.new_empty("roi-by-value/omx-alg")
                param.set_value("delta-qp", -6)
                meta.add_param(param)
                if meta.get_param("roi-by-value/omx-alg").get_value("delta-qp") != -6:
                    raise RuntimeError("Gst ROI binding unavailable")
            pipelines[strategy] = texts
        finally:
            for endpoint in endpoints:
                endpoint.close()
    dpu = Dpu(root, manifest, args.timeout)
    budget = memory_budget(sum((root/p).stat().st_size for p in manifest["files"] if p.startswith("models/yolov5n/")),
                           (root/"models/rl/ddqn_policy.npz").stat().st_size)
    free = available_memory()
    # DPU was created already: its allocations are covered again conservatively.
    if free < sum(budget.values()):
        raise RuntimeError("insufficient DDR: need %d bytes, MemAvailable=%d" % (sum(budget.values()), free))
    report = {"status": "passed_static_preflight", "hardware_smoke_required": True,
              "preflight_model_load_s": time.perf_counter()-started, "swap_enabled": False,
              "policy_replay": {"states": len(golden), "max_q_error": max(q_errors)},
              "memory_budget": budget, "mem_available": free, "kernel": os.uname().release,
              "gstreamer": Gst.version_string(), "encoder_inspect": inspector,
              "dpu": dpu.tensor_report, "pipelines": pipelines, "policy_dataset_sha256": policy.dataset_sha256}
    report["ingress_pipeline"] = ingress_text()
    return manifest, policy, dpu, report


def preload(root, manifest, dpu, timeout):
    """Read all payloads before timers; XModel is deserialized from a tmpfs copy."""
    start = time.perf_counter()
    frame_data = bytearray(FRAMES*FRAME_BYTES)
    with (root / manifest["video"]["path"]).open("rb") as handle:
        view, offset = memoryview(frame_data), 0
        while offset < len(frame_data):
            count = handle.readinto(view[offset:])
            if not count:
                raise RuntimeError("short video preload")
            offset += count
        if handle.read(1):
            raise RuntimeError("oversized video preload")
    if __import__("hashlib").sha256(frame_data).hexdigest() != manifest["files"][manifest["video"]["path"]]["sha256"]:
        raise RuntimeError("preloaded video checksum mismatch")
    return frame_data, time.perf_counter()-start


def run_once(data, bandwidth, policy, dpu, args, strategy, count=FRAMES, smoke=False):
    scaler_text, encoder_text = pipeline_texts(strategy, args.device)
    scaler, encoder = Endpoint(scaler_text, args.timeout), Endpoint(encoder_text, args.timeout)
    ingress = Endpoint(ingress_text(), args.timeout)
    output = io.BytesIO()
    frames, gops, timings, latencies = [], [], [], []
    previous, output_limit = 0., 64*1024*1024
    vcu = encoder.pipeline.get_by_name("vcu")
    start = time.perf_counter()
    ingress.start()
    scaler.start()
    encoder.start()
    try:
        for k in range(count//GOP):
            raw, arrival, ingress_ms = [], [], []
            # Camera-like arrival pacing measures the one-GOP lookahead explicitly.
            for index in range(k*GOP, (k+1)*GOP):
                if args.mode == "paced" and not smoke:
                    delay = start+(index+1)/FPS-time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                admitted = time.perf_counter()
                arrival.append(admitted)
                ingress.push(memoryview(data)[index*FRAME_BYTES:(index+1)*FRAME_BYTES], index)
                raw.append(ingress.nv12(ingress.pull(), index))
                ingress_ms.append((time.perf_counter()-admitted)*1000)
            objects, frame_logs = [], []
            buffered = time.perf_counter()
            for j, payload in enumerate(raw):
                index = k*GOP+j
                t = time.perf_counter()
                scaler.push(payload, index)
                rgb = scaler.rgb(scaler.pull(), index)
                scaled = time.perf_counter()
                detections, stages = dpu.infer(rgb)
                boxes = roi_boxes(detections)
                if smoke:
                    # A fixed synthetic ROI exercises the metadata path even on empty detections.
                    boxes = [{"class_id": 0, "x": 160, "y": 160, "width": 640, "height": 640}]
                objects.append((detections, boxes))
                stages["scaler_ms"] = (scaled-t)*1000
                stages["ingress_ms"] = ingress_ms[j]
                timings.append(stages)
                frame_logs.append({"frame_idx": index, "gop_idx": k, "pts_ns": index*1_000_000_000//FPS,
                                   "detections": detections.tolist(), "roi": boxes, "stages": stages,
                                   "arrival_s": arrival[j]-start, "gop_buffer_ready_s": buffered-start,
                                   "detection_done_s": time.perf_counter()-start})
            score = float(np.mean([semantic(d) for d, _ in objects]))
            before = time.perf_counter()
            selected = decision(strategy, k, bandwidth, previous, score, policy)
            controller_ms = (time.perf_counter()-before)*1000
            target = int(round(selected["target_bitrate_mbps"]*1000))
            vcu.set_property("target-bitrate", target)
            if vcu.get_property("target-bitrate") != target:
                raise RuntimeError("encoder rejected bitrate property")
            selected["set_target_bitrate_kbps"] = target
            if smoke and strategy == "rl":
                # Force a real dynamic update during smoke, separately from policy scoring.
                target = 1000 if k == 0 else 2000
                vcu.set_property("target-bitrate", target)
                if vcu.get_property("target-bitrate") != target:
                    raise RuntimeError("dynamic bitrate smoke failed")
                selected["set_target_bitrate_kbps"] = target
            encode_start = time.perf_counter()
            for j, payload in enumerate(raw):
                roi = objects[j][1] if strategy in ("fixed_roi", "rl") else ()
                frame_logs[j]["appsrc_push_s"] = time.perf_counter()-start
                encoder.push(payload, k*GOP+j, roi, selected["delta_qp_roi"])
            sizes = []
            for j in range(GOP):
                sample = encoder.pull()
                buf = sample.get_buffer()
                ok, mapped = buf.map(encoder.Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError("cannot map H.264 AU")
                try:
                    payload = bytes(mapped.data)
                finally:
                    buf.unmap(mapped)
                key = validate_au(payload, k*GOP+j, buf.pts)
                if output.tell()+len(payload) > output_limit:
                    raise RuntimeError("RAM output limit exceeded")
                output.write(payload)
                sizes.append(len(payload))
                ended = time.perf_counter()
                latency = (ended-arrival[j])*1000
                scheduled_latency = (ended-(start+(k*GOP+j+1)/FPS))*1000 if args.mode == "paced" and not smoke else None
                latencies.append(latency)
                frame_logs[j].update({"encoded_bytes": len(payload), "idr": key, "encoded_done_s": ended-start,
                                      "latency_ms": latency, "scheduled_latency_ms": scheduled_latency})
            actual = sum(sizes)*8/1e6
            selected.update({"gop_idx": k, "bandwidth_mbps": float(bandwidth[k]), "previous_actual_bitrate_mbps": previous,
                             "actual_bitrate_mbps": actual, "encoded_bytes": sum(sizes), "frames": GOP,
                             "controller_ms": controller_ms, "vcu_gop_ms": (time.perf_counter()-encode_start)*1000,
                             "evaluation": k >= 4})
            previous = actual
            gops.append(selected)
            frames.extend(frame_logs)
        encoded_elapsed = time.perf_counter()-start
        ingress.eos()
        scaler.eos()
        encoder.eos()
    finally:
        ingress.close()
        scaler.close()
        encoder.close()
    elapsed = time.perf_counter()-start
    summary = {"status": "passed", "strategy": strategy, "mode": args.mode,
               "frames": count, "gops": count//GOP, "elapsed_s": elapsed,
               "encoded_elapsed_s": encoded_elapsed, "fps": count/encoded_elapsed,
               "latency_ms": {str(p): float(np.percentile(latencies, p)) for p in (50, 95, 99)},
               "scored_latency_ms": ({str(p): float(np.percentile(latencies[4*GOP:], p)) for p in (50, 95, 99)}
                                     if count > 4*GOP else None),
               "mean_stage_ms": {name: float(np.mean([row[name] for row in timings])) for name in timings[0]},
               "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               "latency_definition": "raw GOP admission to encoded AU; includes GOP wait; paced schedule latency also logged",
               "hardware_smoke": smoke}
    return output.getvalue(), frames, gops, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--mode", choices=("paced", "throughput"), default="paced")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.timeout <= 0 or args.repeats < 1:
        parser.error("timeout/repeats must be positive")
    root = args.root.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + args.mode
    destination = root / "results" / run_id
    try:
        manifest, policy, dpu, report = preflight(root, args)
        if args.preflight:
            write_json(destination / "preflight.json", report)
            print(json.dumps({"status": report["status"], "report": str(destination/"preflight.json")}))
            return
        bandwidth = read_trace(root, manifest)
        data, preload_s = preload(root, manifest, dpu, args.timeout)
        # Touch all weights/code paths before timed runs, without reading SD during runs.
        policy.q_values(np.zeros(6))
        warmed = {}
        for strategy in STRATEGIES:
            payload, _, _, summary = run_once(data, bandwidth, policy, dpu, args, strategy, count=60, smoke=True)
            warmed[strategy] = {**summary, "encoded_sha256": __import__("hashlib").sha256(payload).hexdigest()}
        # Require observable effect of synthetic ROI in VBR, not just successful attachment.
        if warmed["vbr"]["encoded_sha256"] == warmed["fixed_roi"]["encoded_sha256"]:
            raise RuntimeError("ROI smoke produced identical VBR bitstream; ROI may be ignored")
        write_json(destination / "smoke.json", warmed)
        if args.smoke_only:
            print("Hardware smoke passed: " + str(destination))
            return
        for strategy in STRATEGIES:
            for repeat in range(args.repeats):
                print("Running %s repeat %d/%d" % (strategy, repeat+1, args.repeats), flush=True)
                payload, frames, gops, summary = run_once(data, bandwidth, policy, dpu, args, strategy)
                directory = destination / strategy / ("repeat_%02d" % (repeat+1))
                directory.mkdir(parents=True, exist_ok=False)
                flush = time.perf_counter()
                with (directory / "output.h264").open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                for name, rows in (("frames.jsonl", frames), ("gops.jsonl", gops)):
                    with (directory / name).open("w") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, allow_nan=False)+"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                summary.update({"preload_s": preload_s, "model_preflight_s": report["preflight_model_load_s"],
                                "flush_sd_s": time.perf_counter()-flush,
                                "output_sha256": sha256(directory/"output.h264")})
                write_json(directory / "summary.json", summary)
                write_json(directory / "manifest.json", {"bundle": manifest, "preflight": report, "repeat": repeat+1,
                                                           "model_trial": True, "sd_transfer_in_timed_run": False})
                del payload, frames, gops
        print("Completed: " + str(destination))
    except Exception as error:
        write_json(destination / "failure.json", {"status": "failed", "error": str(error), "fallback": False})
        raise


if __name__ == "__main__":
    main()
