# YOLOv5 Nano / x264 VCU surrogate dataset pipeline

Tài liệu chi tiết: [Giải thích repo và các quyết định khi tự xây dựng](docs/GIAI_THICH_REPO.md),
bao gồm nguồn bandwidth, cách quy đổi trace, ROI, encode, reward, DDQN và phần cần làm khi triển khai thật.

Benchmark bốn nhóm chất lượng ROI/non-ROI, hiệu quả bitrate, bảo toàn detection và thích ứng bandwidth:
[hướng dẫn và định nghĩa metric](docs/BENCHMARK.md). Sau khi đã có dataset và policy, chạy
`make benchmark`; kết quả ở `reports/benchmark/REPORT.md`. Mặc định metric bổ sung lấy
3 frame/GOP trên toàn bộ 32 action; `make benchmark BENCHMARK_FRAMES=30` đo mọi frame.

Pipeline này tạo metadata và encode grid mức GOP cho offline RL. Cấu hình chuẩn nằm ở
`config/pipeline.yaml`: YOLOv5n ở kích thước inference 640, 1080p30, GOP 30,
H.264 IDR/P, 8 target bitrate và 4 ROI delta-QP (32 action).

## Điều kiện bắt buộc

- NVIDIA GPU và driver hoạt động trên host.
- Docker Engine.
- NVIDIA Container Toolkit. Thiếu bất kỳ dependency bắt buộc nào sẽ dừng pipeline ngay;
  không thay bằng CPU, model, codec, metric hoặc nguồn dữ liệu khác.

Cài toolkit một lần trên host (cần mật khẩu sudo):

```bash
bash scripts/install_nvidia_container_toolkit.sh
```

Script khóa đồng bộ bốn package NVIDIA Container Toolkit ở phiên bản `1.20.1-1`,
cấu hình Docker runtime rồi chạy `nvidia-smi` trong CUDA container để xác minh.

## Chạy

```bash
make build
make preflight
make all
```

Có thể chạy/resume từng bước bằng `make download`, `make normalize`, `make infer`,
`make encode`, `make metrics`, `make traces`, `make transitions`, `make validate`,
`make train`, `make evaluate`, `make export-policy`, `make render-policy`.
Artifact đã hoàn tất chỉ được dùng lại nếu hash cấu hình và input khớp; mismatch làm pipeline dừng.

Phần encoder ROI/libvmaf có thể được kiểm thử độc lập bằng `make build-test && make test-native`;
target này không chạy detector và không thay thế kiểm tra CUDA bắt buộc của pipeline đầy đủ.

Action schedule tuỳ chỉnh là CSV có các cột `video_id,trace_id,gop_idx,action_id`:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/workspace" -w /workspace vcu-surrogate:0.1.0 \
  transitions --schedule config/actions.csv
```

## Output chính

- `artifacts/metadata/detections.parquet` và `gop_semantics.parquet`
- `artifacts/encodes/action_space.parquet`
- `artifacts/metrics/gop_action_grid.parquet`
- `data/processed/bandwidth_traces.parquet`
- `data/processed/transitions.parquet` và `transition_schema.json`
- `reports/validation_report.json`
- `artifacts/rl/ddqn_seed_{13,29,47}.pt` và `.npz` (policy NumPy cho ARM)
- `artifacts/rl/ddqn_selected.npz` (checkpoint chọn theo validation)
- `reports/rl/learning_seed_*.csv`, `evaluation_seed_*.json`, `evaluation_summary.json`
- `artifacts/policy_videos/seed_<selected_seed>/*_Verizon_start0.mp4`: video phát lại
  action của policy được chọn, một file cho mỗi video; mỗi GOP được stream-copy từ
  encode grid, 4 GOP đầu dùng warm-up action 4.

`R_actual` được tính từ payload packet H.264 trên đúng cửa sổ 1 giây. VMAF,
PSNR/SSIM và ROI PSNR/SSIM đều được lưu trước khi reward được tính, nên reward có thể
được tái tạo với trọng số khác mà không encode lại.

Bandwidth dùng các trace LTE uplink `.up` của Mahimahi/Sprout ở commit đã pin.
Train dùng ATT và T-Mobile, validation dùng ATT 2016, test giữ lại Verizon;
trace `*-short.up` chỉ ghi trong catalog vì có thể trùng với trace dài. DDQN lấy
4 giá trị bandwidth đã hoàn tất, bitrate thực tế GOP trước và semantic score hiện tại.
Bốn GOP đầu dùng action 4 (1 Mbps, ROI ΔQP 0) để khởi tạo trạng thái.

Policy `.npz` được đọc bằng `vcu_surrogate.policy_numpy.NumpyPolicy` trên ARM; mô-đun
này chỉ cần NumPy. Gọi `act([BW[k-4], BW[k-3], BW[k-2], BW[k-1],
R_actual[k-1], S[k]])` để lấy `(action_id, target_bitrate_mbps, delta_qp_roi)`.
Các bandwidth dùng Mbps và `S[k]` nằm trong [0,1]. Metadata trong `.npz` chứa
action table, chuẩn hóa state và warm-up contract để kiểm tra nhất quán khi triển khai.

Video trong `policy_videos` là bản xem kết quả action theo trace Verizon từ giây 0.
Chúng được ghép từ GOP đã encode độc lập theo từng action trong grid, không phải
luồng encode liên tục với rate-control thay đổi động theo policy. Vì vậy hãy dùng
chúng để kiểm tra hình ảnh/quyết định offline; đo domain gap và chất lượng triển khai
cuối cùng vẫn cần chạy policy cùng VCU thật trên ZCU106.
