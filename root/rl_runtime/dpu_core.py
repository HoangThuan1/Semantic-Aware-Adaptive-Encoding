"""Vendored pure tensor/YOLO routines; original sha256: 43a1b897740cd5f861811436d5a8a4af8e9234c58aeb017b7708c53c72b3243d"""
import numpy as np

def box_iou_one_to_many(box, boxes):
    if boxes.size == 0:
        return np.empty((0), dtype=np.float32)

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))

    union = area_a + area_b - inter + 1e-9
    return inter / union

def decode_head(head, anchors, input_w, input_h, conf_thres):
    """
    head: [H,W,3,no], raw YOLOv5 logits
    anchors: [3,2] in input pixel units
    output: [N,6] = x1,y1,x2,y2,score,class_id in model input coordinates
    """
    h, w, na, no = head.shape
    if na != 3:
        raise RuntimeError(f"Expected 3 anchors/head, got {na}")

    stride_x = float(input_w) / float(w)
    stride_y = float(input_h) / float(h)

    # final_score = sigmoid(objectness) * sigmoid(class). Since class <= 1,
    # locations whose objectness is already below the final threshold can
    # never survive. Filter those first so a P6 160x160 head does not run
    # sigmoid over all 80 classes at every anchor.
    objectness = sigmoid(head[..., 4])
    candidate_mask = objectness >= conf_thres
    if not np.any(candidate_mask):
        return np.empty((0, 6), dtype=np.float32)

    grid_y, grid_x, anchor_index = np.nonzero(candidate_mask)
    pred = sigmoid(head[candidate_mask])
    scores_all = pred[:, 4:5] * pred[:, 5:]
    cls_id = np.argmax(scores_all, axis=1)
    score = np.max(scores_all, axis=1)
    keep = score >= conf_thres
    if not np.any(keep):
        return np.empty((0, 6), dtype=np.float32)

    pred = pred[keep]
    grid_x = grid_x[keep].astype(np.float32)
    grid_y = grid_y[keep].astype(np.float32)
    anchor_index = anchor_index[keep]
    score_sel = score[keep]
    cls_sel = cls_id[keep].astype(np.float32)

    xy_sel = pred[:, 0:2] * 2.0 - 0.5
    xy_sel[:, 0] = (xy_sel[:, 0] + grid_x) * stride_x
    xy_sel[:, 1] = (xy_sel[:, 1] + grid_y) * stride_y
    wh_sel = (pred[:, 2:4] * 2.0) ** 2 * anchors[anchor_index]

    x1 = xy_sel[:, 0] - wh_sel[:, 0] / 2.0
    y1 = xy_sel[:, 1] - wh_sel[:, 1] / 2.0
    x2 = xy_sel[:, 0] + wh_sel[:, 0] / 2.0
    y2 = xy_sel[:, 1] + wh_sel[:, 1] / 2.0

    return np.stack((x1, y1, x2, y2, score_sel, cls_sel), axis=1).astype(np.float32)

def dequantize_output(raw, tensor):
    fix = tensor_fix_point(tensor)
    if fix is None:
        return raw.astype(np.float32, copy=False)
    return raw.astype(np.float32) * float(2 ** (-fix))

def get_dpu_subgraphs(graph):
    root = graph.get_root_subgraph()
    result = []

    def walk(sg):
        if sg.has_attr("device"):
            try:
                device = str(sg.get_attr("device")).upper()
            except Exception:
                device = ""
            if device == "DPU":
                result.append(sg)
                return
        for child in sg.toposort_child_subgraph():
            walk(child)

    walk(root)
    return result

def infer_input_layout(shape):
    if len(shape) != 4:
        raise RuntimeError(f"Expected 4-D model input, got {shape}")

    if shape[-1] in (1, 3, 4):
        # N,H,W,C
        return "NHWC", shape[1], shape[2], shape[3]
    if shape[1] in (1, 3, 4):
        # N,C,H,W
        return "NCHW", shape[2], shape[3], shape[1]

    raise RuntimeError(f"Cannot infer input layout from shape {shape}. " "Expected NHWC or NCHW image tensor.")

def match_anchors_to_head(head, all_anchors, input_w, input_h):
    """
    Pick anchor group by nearest conventional YOLOv5 stride.
    This avoids relying on VART output tensor ordering.
    """
    h, w = head.shape[:2]
    sx = input_w / float(w)
    sy = input_h / float(h)
    stride = (sx + sy) / 2.0

    conventional = 8.0 * (2.0 ** np.arange(len(all_anchors), dtype=np.float32))
    idx = int(np.argmin(np.abs(conventional - stride)))
    return idx, stride, all_anchors[idx].copy()

def normalize_yolo_head(raw, num_classes):
    """
    Return a YOLO detection head as [H, W, 3, no], where no=nc+5.

    Supported common layouts:
      [1,H,W,3*no]
      [1,3*no,H,W]
      [1,H,W,3,no]
      [1,3,H,W,no]
      [1,3,no,H,W]
    """
    no = num_classes + 5
    x = np.asarray(raw)

    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    # [H,W,3*no]
    if x.ndim == 3 and x.shape[-1] == 3 * no:
        h, w, _ = x.shape
        return x.reshape(h, w, 3, no)

    # [3*no,H,W]
    if x.ndim == 3 and x.shape[0] == 3 * no:
        x = np.transpose(x, (1, 2, 0))
        h, w, _ = x.shape
        return x.reshape(h, w, 3, no)

    if x.ndim == 4:
        # [H,W,3,no]
        if x.shape[2] == 3 and x.shape[3] == no:
            return x

        # [3,H,W,no]
        if x.shape[0] == 3 and x.shape[3] == no:
            return np.transpose(x, (1, 2, 0, 3))

        # [3,no,H,W]
        if x.shape[0] == 3 and x.shape[1] == no:
            return np.transpose(x, (2, 3, 0, 1))

        # [H,W,no,3]
        if x.shape[2] == no and x.shape[3] == 3:
            return np.transpose(x, (0, 1, 3, 2))

    raise RuntimeError(f"Unsupported YOLO output shape {tuple(raw.shape)} for " f"num_classes={num_classes} (expected no={no}).")

def sigmoid(x):
    # Clip avoids exp overflow for strongly quantized logits.
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def tensor_dims(tensor):
    # Python VART/XIR versions expose dims consistently enough for this.
    return tuple(int(x) for x in tensor.dims)

def tensor_fix_point(tensor):
    try:
        if tensor.has_attr("fix_point"):
            return int(tensor.get_attr("fix_point"))
    except Exception:
        pass
    return None
