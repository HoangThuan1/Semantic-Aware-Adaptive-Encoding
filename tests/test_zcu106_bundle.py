"""Host contract tests; these do not stand in for actual DPU/VCU smoke tests."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "root"
sys.path.insert(0, str(ROOT))
from rl_runtime.common import (CLASS_WEIGHTS, FRAME_BYTES, FRAMES, GOP, STRATEGIES,
                               annexb_nal_types, bundle_manifest, decision, memory_budget,
                               read_trace, roi_boxes, semantic, validate_au)
from rl_runtime.policy_numpy import NumpyPolicy
from rl_runtime.board import (Dpu, Endpoint, global_nms, pipeline_texts, ingress_text,
                              resize_rgb_for_dpu, BOARD_PREPROCESSING)
from rl_runtime import dpu_core


@pytest.fixture
def policy():
    return NumpyPolicy(ROOT / "models/rl/ddqn_policy.npz")


def test_packaged_test_segment_and_policy(policy):
    manifest = bundle_manifest(ROOT)
    assert manifest["video"]["split"] == "test"
    assert manifest["video"]["video_id"] == "bdd_114af10f-c3d93b85"
    assert manifest["detector"]["board_preprocessing"] == BOARD_PREPROCESSING
    assert manifest["trace"]["split"] == "test"
    assert manifest["trace"]["offset_gop"] == 0
    assert read_trace(ROOT, manifest).shape == (32,)
    assert policy.action_table.shape == (32, 2)
    assert FRAMES*FRAME_BYTES == 2985984000
    assert GOP*FRAME_BYTES == 93312000


def test_strategy_contract_and_feedback(policy):
    bw = np.arange(10, dtype=float)+1
    for strategy in STRATEGIES:
        row = decision(strategy, 0, bw, 0, .25, policy)
        assert row["target_bitrate_mbps"] == (1 if strategy == "rl" else 1.5)
        assert row["delta_qp_roi"] == (-6 if strategy == "fixed_roi" else 0)
    row = decision("rl", 4, bw, 1.23, .25, policy)
    assert row["raw_state"] == [1, 2, 3, 4, 1.23, .25]
    assert row["action_id"] == int(np.argmax(row["q_values"]))
    assert decision("rl", 3, bw, 3, .9, policy)["action_id"] == 4
    with pytest.raises(ValueError):
        decision("unknown", 0, bw, 0, 0, policy)


def test_roi_padding_alignment_semantic():
    boxes = np.array([[0, 0, 1920, 1080, 1, 0], [160, 160, 320, 320, .5, 2]])
    rois = roi_boxes(boxes)
    assert rois[0] == {"class_id": 0, "x": 0, "y": 0, "width": 1920, "height": 1080}
    assert rois[1]["x"] == 144 and rois[1]["width"] == 192
    expected = 1-np.exp(-(1+.8*.5*np.sqrt(160*160/(1920*1080))))
    assert semantic(boxes) == pytest.approx(expected)
    assert semantic(np.empty((0, 6))) == 0


def test_global_nms_does_not_starve_high_confidence_classes():
    boxes = np.array([[0, 0, 10, 10, .5, 0], [0, 0, 10, 10, .9, 2],
                      [1, 1, 9, 9, .8, 2]], dtype=np.float32)
    result = global_nms(dpu_core, boxes, .45, 1)
    assert len(result) == 1 and result[0, 5] == 2


def test_nal_pts_and_idr_fail_closed():
    key = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x65hello"
    predicted = b"\x00\x00\x01\x41hello"
    assert annexb_nal_types(key) == [7, 5]
    assert validate_au(key, 0, 0)
    assert not validate_au(predicted, 1, 33333333)
    with pytest.raises(RuntimeError, match="IDR"):
        validate_au(predicted, 30, 1000000000)
    with pytest.raises(RuntimeError, match="PTS"):
        validate_au(predicted, 1, 100)
    with pytest.raises(RuntimeError, match="header-only"):
        validate_au(b"\x00\x00\x01\x67abc", 0, 0)


def test_pipeline_has_hardware_only_paths_and_bounded_queues():
    assert "appsrc" in ingress_text() and "appsink" in ingress_text()
    assert "format=NV12" in ingress_text()
    for strategy in STRATEGIES:
        scaler, encoder = pipeline_texts(strategy, "/dev/video0")
        assert "vvas" not in scaler+encoder
        assert "v4l2convert" in scaler and "capture-io-mode=4 output-io-mode=4" in scaler
        assert "width=1280,height=720" in scaler
        assert "xlnxabrscaler" not in scaler
        assert "interlace-mode=progressive" in encoder and "chroma-site=mpeg2" in encoder
        assert "omxh264enc" in encoder and "gop-length=30 periodicity-idr=30" in encoder
        assert "max-buffers=30 drop=false" in encoder
        assert "qp-mode=roi" in encoder if strategy in ("rl", "fixed_roi") else "qp-mode=uniform" in encoder
    with pytest.raises(ValueError):
        pipeline_texts("rl", '/dev/video0" ! identity')


def test_budget_counts_input_gop_output_and_reserves():
    budget = memory_budget(100, 200)
    assert budget["input"] == 2985984000
    assert budget["gop_buffers"] == 186624000
    assert sum(budget.values()) > budget["input"] + budget["gop_buffers"]


def test_independent_golden_replay(policy):
    path = ROOT / "models/rl/golden_policy.json"
    if not path.exists():
        pytest.skip("offline CUDA reference has not yet been generated")
    rows = json.loads(path.read_text())
    assert [r["gop_idx"] for r in rows] == list(range(4, 32))
    for row in rows:
        q = policy.q_values(row["raw_state"])[0]
        np.testing.assert_allclose(q, row["q_values"], rtol=1e-5, atol=1e-5)
        assert int(q.argmax()) == row["action_id"]


def test_missing_frame_and_encoder_bus_errors_fail(monkeypatch):
    import rl_runtime.board as board
    endpoint = Endpoint.__new__(Endpoint)
    endpoint.timeout = .5
    endpoint.Gst = SimpleNamespace(MessageType=SimpleNamespace(ERROR=1))
    endpoint.pipeline = SimpleNamespace(get_bus=lambda: SimpleNamespace(pop_filtered=lambda _: None))
    endpoint.sink = SimpleNamespace(emit=lambda *_: None, get_property=lambda _: False)
    clock = iter([0., .1, .3, .6])
    monkeypatch.setattr(board.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="timeout"):
        endpoint.pull()
    endpoint.sink.get_property = lambda _: True
    clock = iter([0., .1])
    with pytest.raises(RuntimeError, match="early appsink EOS"):
        endpoint.pull()
    message = SimpleNamespace(parse_error=lambda: ("VCU failed", "driver error"))
    endpoint.pipeline = SimpleNamespace(get_bus=lambda: SimpleNamespace(pop_filtered=lambda _: message))
    with pytest.raises(RuntimeError, match="VCU failed"):
        endpoint.check_bus()


def test_scaler_rgb_stride_and_pts(monkeypatch):
    endpoint = Endpoint.__new__(Endpoint)
    endpoint.Gst = SimpleNamespace(MapFlags=SimpleNamespace(READ=1))
    offset, stride = 16, 3904
    data = bytearray(offset+720*stride)
    expected = np.arange(720*1280*3, dtype=np.uint8).reshape(720, 1280, 3)
    target = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=data, offset=offset, strides=(stride, 3, 1))
    target[:] = expected
    info = SimpleNamespace(width=1280, height=720, finfo=SimpleNamespace(name="RGB"), stride=[stride], offset=[offset])
    endpoint.GstVideo = SimpleNamespace(VideoInfo=SimpleNamespace(new_from_caps=lambda _: info),
                                       buffer_get_video_meta=lambda _: None)
    buffer = SimpleNamespace(pts=33333333, map=lambda _: (True, SimpleNamespace(data=data)), unmap=lambda _: None)
    sample = SimpleNamespace(get_buffer=lambda: buffer, get_caps=lambda: None)
    np.testing.assert_array_equal(endpoint.rgb(sample, 1), expected)
    buffer.pts = 0
    with pytest.raises(RuntimeError, match="PTS"):
        endpoint.rgb(sample, 1)


def test_required_arm_resize_contract(monkeypatch):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    expected = np.full((360, 640, 3), 42, dtype=np.uint8)
    calls = []
    def resize(rgb, size, interpolation):
        calls.append((rgb.shape, size, interpolation))
        return expected
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(resize=resize, INTER_LINEAR=1))
    assert resize_rgb_for_dpu(image) is expected
    assert calls == [((720, 1280, 3), (640, 360), 1)]
    with pytest.raises(RuntimeError, match="1280x720"):
        resize_rgb_for_dpu(np.zeros((360, 640, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="uint8"):
        resize_rgb_for_dpu(image.astype(np.float32))


@pytest.mark.parametrize("layout", ["NHWC", "NCHW"])
def test_dpu_input_uses_resized_rgb_and_centered_letterbox(monkeypatch, layout):
    resized = np.empty((360, 640, 3), dtype=np.uint8)
    resized[:] = [10, 20, 30]
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        resize=lambda *args, **kwargs: resized, INTER_LINEAR=1))
    dpu = Dpu.__new__(Dpu)
    dpu.layout = layout
    dpu.shape = (1, 640, 640, 3) if layout == "NHWC" else (1, 3, 640, 640)
    dpu.lut = np.arange(256).astype(np.int8)
    dpu.timeout, dpu.outputs = 1., []
    captured = []
    def execute(inputs, outputs):
        captured.append(inputs[0])
        return 1
    dpu.runner = SimpleNamespace(execute_async=execute, wait=lambda *args: 5)
    with pytest.raises(RuntimeError, match="DPU timeout/error"):
        dpu.infer(np.zeros((720, 1280, 3), dtype=np.uint8))
    packed = captured[0]
    assert packed.dtype == np.int8 and packed.flags.c_contiguous
    image = packed[0] if layout == "NHWC" else packed[0].transpose(1, 2, 0)
    assert np.all(image[:140] == 114) and np.all(image[500:] == 114)
    np.testing.assert_array_equal(image[140:500], resized.astype(np.int8))


def test_ingress_nv12_stride_and_pts():
    endpoint = Endpoint.__new__(Endpoint)
    endpoint.Gst = SimpleNamespace(MapFlags=SimpleNamespace(READ=1))
    stride = 1952
    offsets = [16, 16+1080*stride]
    data = bytearray(offsets[1]+540*stride)
    for offset, height, value in ((offsets[0], 1080, 90), (offsets[1], 540, 128)):
        np.ndarray((height, 1920), dtype=np.uint8, buffer=data, offset=offset, strides=(stride, 1))[:] = value
    info = SimpleNamespace(width=1920, height=1080, finfo=SimpleNamespace(name="NV12"),
                           stride=[stride, stride], offset=offsets)
    endpoint.GstVideo = SimpleNamespace(VideoInfo=SimpleNamespace(new_from_caps=lambda _: info),
                                       buffer_get_video_meta=lambda _: None)
    buffer = SimpleNamespace(pts=0, map=lambda _: (True, SimpleNamespace(data=data)), unmap=lambda _: None)
    sample = SimpleNamespace(get_buffer=lambda: buffer, get_caps=lambda: None)
    result = endpoint.nv12(sample, 0)
    assert len(result) == FRAME_BYTES and result[0] == 90 and result[1920*1080] == 128
    buffer.pts = 33333333
    with pytest.raises(RuntimeError, match="PTS"):
        endpoint.nv12(sample, 0)
