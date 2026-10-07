# ZCU106 — Vivado flow, GStreamer không VVAS

Copy nguyên `root/` sang `/mnt/sdroot/root/`. Không cần chuyển code về đường dẫn
khác, không chạy VVAS và không thay bitstream/rootfs đang hoạt động.

```bash
cd /mnt/sdroot/root
bash tools/preflight.sh
python3 tools/replay_policy.py
bash vcu_test/smoke.sh
bash tools/run_all.sh
# Một bộ riêng đo throughput cực đại, cùng dữ liệu và bốn chiến lược:
bash tools/run_all.sh --mode throughput
```

Mặc định scaler là `/dev/video0`; dùng `--device /dev/videoN` để chọn đúng thiết
bị đã xác minh, không tự dò/đổi thiết bị. Mặc định ba lần mỗi chiến lược,
timeout 30 giây. Thiếu dependency, RAM, model/tensor mismatch hoặc timeout làm
runtime dừng, ghi `results/<run_id>/failure.json`. Không fallback.

Preflight yêu cầu NumPy, OpenCV (`cv2`), Python GI Gst/GstApp/GstVideo, VART/XIR, `xdputil`,
`gst-inspect-1.0`, Multi-Scaler V4L2 và `omxh264enc`/`h264parse`. Các plugin
phải hỗ trợ NV12 bt709 → RGB sRGB 1280×720, input XModel 640×640 INT8,
ROI `roi-by-value/omx-alg:delta-qp`, và target-bitrate mutable ở PLAYING.
Preflight tĩnh chưa chứng minh hardware chạy; `smoke.sh` và warm-up trong
`run_all.sh` chạy DPU/scaler/encoder thật và thử ROI/dynamic bitrate.

Input là 32 giây đầu của bdd_114af10f-c3d93b85, thuộc split test của project rl:
960 frame NV12 1080p30, 32 GOP, **2.985.984.000 byte ≈ 3 GB raw**.
Trace test thật đủ 32 GOP theo manifest. Manifest khóa hash dữ liệu/code,
trace offset/timestamp, policy trial và XModel. Checkpoint thiếu dataset hash;
không coi là đã được train trên dataset test hiện tại. Chưa xác minh XModel
cùng weights với `.pt`: không quy toàn bộ chênh lệch YOLO cho INT8.

Runtime preload input vào bytearray DDR, nạp policy/trace và deserialize DPU
trước khi đo. Model/routines được warm-up trước đo. GStreamer appsrc/appsink
đưa NV12 từ RAM vào GOP buffer; hai pipeline appsrc/appsink tiếp theo thực hiện
scaler và encoder. Controller giữ raw GOP trong RAM và ghép RGB/detection bằng
PTS. Không có software encoder hay CPU detector.
Đường cố định theo pipeline đã chạy được trên board: `v4l2convert
capture-io-mode=4 output-io-mode=4` → RGB 1280×720 → ARM/OpenCV
`INTER_LINEAR` resize 640×360 → letterbox 640×640 (pad 114, trên/dưới 140)
→ VART/DPU. ARM resize là preprocessing bắt buộc, không phải fallback;
thiếu OpenCV sẽ dừng. `arm_resize_ms` được ghi riêng trong timing.
Không sử dụng `xlnxabrscaler`, không thử đổi sang scaler caps khác khi lỗi.
Nguồn DDR `appsrc` đã push đúng một frame mỗi buffer cùng PTS theo frame index,
nên không cần `rawvideoparse`; nguồn `filesrc` đọc raw mới cần parser này.
Mỗi GOP 93.312.000 byte; input 2.985.984.000 byte. Preflight cộng thêm buffer,
output tối đa 64 MiB, runtime/hardware reserve 256 MiB và safety 256 MiB.

Preflight yêu cầu swap tắt để anonymous RAM không bị swap xuống SD.
Output/log trong RAM khi đo; fsync video/log xuống SD sau mỗi lần. `paced`
mô phỏng nhịp frame 30 Hz; `scheduled_latency_ms` còn tính thời gian trễ do
pipeline không theo kịp nhịp này. `latency_ms` bắt đầu lúc frame được nhận vào
GOP buffer. `throughput` không sleep. Thời gian trace luôn theo PTS, không
theo wall clock. Peak RSS là peak của process cả bộ chạy, không phải công suất
hay lượng DDR của PL. Không có đo điện nếu chưa có sensor thực.

CBR: 1,5 Mbps, constant, không ROI. VBR: 1,5 Mbps, variable, không ROI.
Fixed ROI: 1,5 Mbps, variable, delta −6. RL: variable, policy chọn bitrate và
delta mỗi GOP; bốn GOP đầu 1 Mbps/delta 0. Mọi chiến lược chạy cùng DPU và
GOP buffering. Bảng chính tính GOP 4–31, bảng phụ toàn clip. BW trace dùng cho
state và đánh giá network queue offline; file output không giả làm mạng thật.

**Board không tính VMAF/ROI SSIM/reward.** `run_all.sh` chỉ chạy pipeline
inference/controller/VCU encode và ghi bitstream, detections, state/action,
bitrate feedback, timing. `compare_results.py` là công cụ **host-only** để tính
VMAF/ROI SSIM/reward sau khi copy kết quả board về máy offline.

## Host tools

Các công cụ host cần CUDA/PyTorch + pinned YOLOv5, FFmpeg libvmaf, pandas,
pyarrow, OpenCV, PyYAML, NumPy, tqdm. Chạy trong Docker image đã có của repo:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -v /home/thuanhxfg/my_pytorch/cnn_training/semantic_training:/semantic:ro \
  --entrypoint python vcu-surrogate:0.1.0-cuda \
  root/tools/prepare_bundle.py --semantic-source /semantic
```

`prepare_bundle.py` chuẩn hóa clip, kiểm tra test split/source checksum, chọn
trace, copy NPZ/XModel và snapshot các hàm YOLO/metrics cần thiết. Không thay
artifact khác hash. Binary helper x264 chỉ dùng trên host (không chạy trên ARM):

```bash
g++ -O2 -std=c++17 root/tools/offline_encoder.cpp -lx264 -o /tmp/zcu106_offline_encoder
python root/tools/offline_reference.py --encoder /tmp/zcu106_offline_encoder
```

Offline x264 CBR dùng HRD/filler; board CBR dùng control-rate constant với
filler tắt. Báo cáo ghi hai cấu hình riêng, không khẳng định rate-control hai
encoder giống hệt. VBR x264 tắt HRD để cho phép reconfigure VBV ở mỗi GOP.
Host x264 dùng một thread để không giữ frame của GOP sau khi cần feedback GOP trước.
Offline reference chạy detector FP32 CUDA, square letterbox 640 giống contract
board, persistent libx264 và policy feedback bitrate thật. Golden Q-values
được kiểm tra bằng PyTorch độc lập. Resize/color conversion host và scaler PL
có thể khác nhau; log comparison không khẳng định preprocessing bit-exact.

Sau khi copy kết quả từ board về đúng `root/results/`, chạy trên host:

```bash
python root/tools/compare_results.py --board-run root/results/<run_id>
# Kiểm tra bitstream/quality offline thật trước khi có kết quả board:
python root/tools/compare_results.py --offline-only
```

So bốn chiến lược offline/board, VMAF/PSNR/SSIM/ROI SSIM với cùng FP32 reference
ROI và reward gốc của rl. Mặc định ROI quality sample stride 5 như project rl;
dùng `--sample-stride 1` để đánh giá mọi frame. Không thay VMAF bằng metric khác.
So action end-to-end có thể lệch do detection/VCU feedback; replay cùng state
kiểm tra policy độc lập. Chỉ báo mAP nếu bổ sung nhãn thật; hiện là detector
agreement với pseudo labels.

Các API timestamp/queue dùng [GStreamer appsrc](https://gstreamer.freedesktop.org/documentation/app/appsrc.html);
ROI sử dụng [GstVideoRegionOfInterestMeta](https://gstreamer.freedesktop.org/documentation/video/gstvideometa.html).
