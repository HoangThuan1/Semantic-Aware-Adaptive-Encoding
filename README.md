# YOLOv5 Nano / x264 VCU surrogate dataset pipeline

Tài liệu chi tiết: [Giải thích repo và các quyết định khi tự xây dựng](docs/GIAI_THICH_REPO.md),
bao gồm nguồn bandwidth, cách quy đổi trace, ROI, encode, reward, DDQN và phần cần làm khi triển khai thật.

## Xem kết quả ngay trên GitHub

- [Báo cáo benchmark](reports/fcc_bdd/benchmark/REPORT.md),
  [biểu đồ so sánh policy](reports/fcc_bdd/benchmark/policy_comparison.png) và
  [bảng tổng hợp](reports/fcc_bdd/benchmark/policy_summary.csv).
- [Kết quả đánh giá RL](reports/fcc_bdd/rl/evaluation_summary.json) và
  [learning curve](reports/fcc_bdd/rl/learning_seed_13.csv).
- [So sánh offline cho bundle ZCU106](root/results/offline_reference/comparison/strategy_summary.csv)
  và [trạng thái xác minh](root/results/offline_reference/comparison/report.json).
  Đây là kết quả host offline; chưa xác minh kết quả chạy trên board.
- [Policy NumPy được chọn](artifacts/rl/ddqn_selected.npz) dùng cho pipeline hiện tại;
  bundle ZCU106 giữ policy snapshot riêng theo `root/manifest.json`.

Git giữ code, tài liệu, biểu đồ, bảng kết quả và model nhỏ của bundle ZCU106.
Video nguồn/raw NV12, encode grid, Parquet, cache và log được giữ cục bộ,
không đi kèm bản clone; tạo lại bằng pipeline và hướng dẫn `root/README.md`
trước khi copy bundle đầy đủ lên SD. Các file Parquet được liệt kê trong báo cáo
là output khi chạy pipeline, không phải file có sẵn trên GitHub.

Benchmark bốn nhóm chất lượng ROI/non-ROI, hiệu quả bitrate, bảo toàn detection và thích ứng bandwidth:
[hướng dẫn và định nghĩa metric](docs/BENCHMARK.md). Sau khi đã có dataset và policy, chạy
`make benchmark`; kết quả ở `reports/fcc_bdd/benchmark/REPORT.md`. Mặc định metric bổ sung lấy
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
- `artifacts/rl/ddqn_seed_13.pt` và `.npz` (policy NumPy cho ARM)
- `artifacts/rl/ddqn_selected.npz` (checkpoint chọn theo validation)
- `reports/fcc_bdd/rl/learning_seed_*.csv`, `evaluation_seed_*.json`, `evaluation_summary.json`
- `artifacts/policy_videos/seed_<selected_seed>/*_<trace_id>_start0.mp4`: video phát lại
  action của policy được chọn, một file cho mỗi video test; mỗi GOP được stream-copy từ
  encode grid, 4 GOP đầu dùng warm-up action 4.

`R_actual` được tính từ payload packet H.264 trên đúng cửa sổ 1 giây. VMAF,
PSNR/SSIM và ROI PSNR/SSIM đều được lưu trước khi reward được tính, nên reward có thể
được tái tạo với trọng số khác mà không encode lại.

Bandwidth dùng FCC và Norway từ MERINA, khóa commit trong config. Trace được chia theo
phiên đo, độc lập train/validation/test 70/15/15. Dữ liệu timestamp/throughput được tích phân
thành Mbps mỗi giây, theo convention throughput dòng i phục vụ khoảng [t[i-1], t[i]).
Train dùng 7 episode trace thật và 3 synthetic trong mỗi block 10 episode; chọn nguồn thật
FCC/Norway 50/50. Synthetic ghép block 10 giây chỉ từ trace train và lưu lineage.

`make download` tải khoảng 1 GB video BDD100K nguyên vẹn qua HTTP Range, kiểm ZIP CRC,
decode và SHA-256. Manifest split gộp video mới với video cũ, chia 70% phát triển/30% test;
10% phần phát triển làm validation (xấp xỉ 63/7/30 toàn tập). Không chia theo frame/GOP.
State normalizer chỉ fit trên video và trace train. Manifest được hash trong artifact/policy.
Trace FCC/Norway chuẩn hóa thành `trace_timestamp_mbps.parquet` (`timestamp_s`, `mbps`),
và tích phân thành `bandwidth_traces.parquet` theo giây (`timestamp_s`, `bandwidth_mbps`).
Outage giữ 0; giây cuối chưa đủ bị bỏ. Dữ liệu thiếu/lỗi hoặc checksum sai làm pipeline dừng, không fallback.
Dữ liệu mới nằm ở `data/{raw,processed}/fcc_bdd`, artifact dùng chung `artifacts/`, báo cáo
ở `reports/fcc_bdd`. Artifact cũ được giữ cùng báo cáo cũ tại `reports/benchmark/legacy_artifacts/`.

DDQN lấy 4 giá trị bandwidth đã hoàn tất, bitrate thực tế GOP trước và semantic score hiện tại.
Bốn GOP đầu dùng action 4 (1 Mbps, ROI ΔQP 0) để khởi tạo trạng thái.

Policy `.npz` được đọc bằng `vcu_surrogate.policy_numpy.NumpyPolicy` trên ARM; mô-đun
này chỉ cần NumPy. Gọi `act([BW[k-4], BW[k-3], BW[k-2], BW[k-1],
R_actual[k-1], S[k]])` để lấy `(action_id, target_bitrate_mbps, delta_qp_roi)`.
Các bandwidth dùng Mbps và `S[k]` nằm trong [0,1]. Metadata trong `.npz` chứa
action table, chuẩn hóa state và warm-up contract để kiểm tra nhất quán khi triển khai.

Video trong `policy_videos` là bản xem kết quả action theo trace test hợp lệ đầu tiên theo ID từ giây 0.
Chúng được ghép từ GOP đã encode độc lập theo từng action trong grid, không phải
luồng encode liên tục với rate-control thay đổi động theo policy. Vì vậy hãy dùng
chúng để kiểm tra hình ảnh/quyết định offline; đo domain gap và chất lượng triển khai
cuối cùng vẫn cần chạy policy cùng VCU thật trên ZCU106.

## Chạy có báo cáo trạng thái

Runner kiểm thử, đo dung lượng reference trên mẫu 6 giây, kiểm tra dung lượng trống rồi
chạy tất cả bước, gồm export và benchmark. Runner chờ downloader đang chạy hoàn tất.

```bash
docker run -d --name vcu-fcc-bdd-pipeline --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/workspace" -w /workspace \
  --entrypoint python vcu-surrogate:0.1.0 scripts/run_fcc_bdd.py
```

Theo dõi `reports/fcc_bdd/run_status.json`, `RUN_REPORT.md` và `logs/<stage>.log`.
Runner dừng và ghi nguyên nhân nếu một bước thất bại; chỉ ghi complete sau benchmark.
Video subset được lấy theo thứ tự archive, không bảo đảm cân bằng thời tiết/thời gian.

Metrics và policy rollout được ghi Parquet từng phần để tránh gom toàn bộ kết quả BDD100K vào RAM. Benchmark kiểm tra hash dataset của từng policy trước khi đánh giá.

Để tiếp tục runner đã dừng: `docker start vcu-fcc-bdd-pipeline`. Các bước đã hoàn tất được giữ theo config/dataset hash; bước đang dở chạy lại và tái sử dụng artifact có sidecar hợp lệ.

Metrics hiện dùng 6 worker CPU, mỗi lượt VMAF dùng 2 thread; cấu hình `cuda_workers: 0` không chạy CUDA VMAF. Chất lượng lấy mẫu mỗi 5 frame (GOP offsets 0, 5, 10, 15, 20, 25), giữ ngữ cảnh chuyển động đầy đủ cho VMAF. GOP vẫn có 30 frame encoded và đủ packet bytes; `metric_sample_count` ghi 6 mẫu chất lượng. Cache metric full-frame đã xác minh có thể chuyển sang lịch lấy mẫu này mà không decode lại.

Bộ thử nghiệm ZCU106 nằm trong [root/README.md](root/README.md). Bundle dùng policy/reward snapshot riêng theo `root/manifest.json`; xem phần runtime trong [tài liệu giải thích repo](docs/GIAI_THICH_REPO.md) trước khi so kết quả với benchmark FCC/BDD hiện tại.

Reward hiện tại phạt vượt BW mạnh trong vùng 1×–1,5× (violation weight 1,0), switching weight 0,05. Benchmark so DDQN, DDQN fixed ROI ΔQP −6, CBR/VBR mô phỏng và optimal offline; tham số CBR/VBR chọn trên validation. Chạy lại bằng `make train`, `make evaluate`, `make benchmark`; theo dõi `reports/fcc_bdd/run_status.json` và `logs/train.log`.
