# Benchmark chất lượng video, bitrate, ngữ nghĩa và thích ứng mạng

Log thay đổi reward và kết quả: [ORACLE_REWARD_LOG.md](../reports/fcc_bdd/benchmark/ORACLE_REWARD_LOG.md).

## Chạy

Cần dataset/grid và policy đã được tạo bằng pipeline hiện có. Benchmark không encode/train lại và không thay đổi các artifact nguồn.

```bash
make benchmark
# Đo bổ sung mọi frame thay cho 3 frame/GOP mặc định:
make benchmark BENCHMARK_FRAMES=30
```

Có thể gọi trực tiếp CLI để chỉnh batch detector:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/workspace" -w /workspace vcu-surrogate:0.1.0 \
  benchmark --frames-per-gop 3 --batch-size 8
```

Đọc kết quả ở `reports/fcc_bdd/benchmark/REPORT.md`. Lần chạy hoàn tất có `manifest.json` với `status=complete`. Cache từng video/action được kiểm input/config/code/output hash; job thiếu hoặc mismatch được tính lại. Có thể chạy lại sau ngắt để dùng các shard đã hoàn tất. Khi thay implementation benchmark, hash code đổi làm cache mất hiệu lực. Không sửa file code trong lúc benchmark đang chạy.

Mặc định đo thêm 3 frame tại offsets `[0,14,29]` trên **mọi GOP và mọi action**. VMAF và ROI metric của grid hiện lấy mẫu stride 5 (6 frame/GOP), VMAF giữ ngữ cảnh chuyển động đầy đủ. Bộ benchmark bao phủ các chiều đánh giá; mặc định không đo detector trên mọi frame. Chọn 30 nếu cần độ phủ thời gian đầy đủ cho metric bổ sung; lựa chọn này không đổi lịch sampling của grid nguồn. Frame sampled của reference tạm được lưu bằng mmap rồi xóa sau mỗi video, hạn chế bộ nhớ và tránh decode reference 32 lần. Batch size chỉ điều khiển detector, không thay đổi lịch lấy mẫu.

## Lần chạy FCC/BDD

`scripts/run_fcc_bdd.py` dùng `scripts/prepare_grid_compact.py`: xử lý lần lượt từng video với đủ 32 action, metrics lấy mẫu mỗi 5 frame; sau mỗi hai video hoàn tất metrics, chạy benchmark 3 frame/GOP cho cặp đó rồi tiếp tục. Video cuối lẻ được benchmark sau khi hết phần local. Giữ FFV1, video nguồn, encodes, manifest/checksum, metric frame/GOP và cache benchmark. Stage metrics tổng hợp cache; stage benchmark cuối dùng lại cache đã có và đo những action còn thiếu; train/evaluate dùng các split và cấu hình đã khóa. Tiến độ nằm trong `reports/fcc_bdd/compact_progress.json`.

Không áp dụng giới hạn dung lượng nhân tạo hoặc dừng chờ compact VDisk. Runner tiếp tục đến train/evaluate/export/render/benchmark. `scripts/watch_fcc_bdd_storage.py` chỉ ghi telemetry dung lượng mỗi 5 giây, không dừng container.

Mỗi reference được decode FFV1 một lần sang YUV420 raw, sau đó chuyển YUV thành BGR cache bằng đúng libswscale mà OpenCV dùng. Sáu worker đọc chung hai cache trên SSD. Mỗi action cũng chỉ decode một lần; từng frame YUV đi qua VMAF để giữ ngữ cảnh chuyển động; chỉ frame mẫu chuyển BGR và tính ROI. Luồng pipe chỉ giữ vài frame trong RAM. Test đối chiếu từng pixel BGR và tất cả metric với cách cũ, yêu cầu kết quả bằng nhau.

Kiểm checksum reference, định dạng và số frame trước khi dùng. Cache đầy đủ có checksum YUV/BGR trong `compact_progress.json` để dùng lại sau restart; cache không khớp được tạo lại từ nguồn đã kiểm tra. Sau metrics, xóa cache YUV/BGR; benchmark tạo cache frame sampled riêng rồi xóa. Xóa JSON VMAF sau khi xác minh checksum và coverage metrics của đủ 32 action; giữ parquet frame/GOP để chạy benchmark sau. FFV1 và encode luôn được giữ. Artifact dùng chung `artifacts/`.

ROI SSIM dùng box filter OpenCV float64 với cùng cửa sổ, sample covariance, K1/K2, pooling ba kênh và bỏ viền như skimage; kiểm thử sai số 1e-10. Metrics dùng 6 worker CPU × 2 thread VMAF, không dùng CUDA VMAF, chung hàng đợi action; lấy mẫu mỗi 5 frame (6 mẫu/GOP), giữ ngữ cảnh VMAF đầy đủ; decoder FFmpeg/ROI 2 luồng, filter graph FFmpeg 1 luồng và OpenCV 1 luồng. Resume encode kiểm SHA-256 bitstream thay vì decode lại video đã xác minh. Resume giữ các metrics đã hoàn tất với đúng provenance.

Thử 60 frame thật dưới tải đang chạy: VMAF FFV1 36.73 s, YUV cache 28.86 s; ROI FFV1 238.90 s, BGR cache 103.16 s. Kết quả từng frame giống hệt; đây là đo mẫu dưới tải, chưa phải tốc độ toàn pipeline. Reference của video chưa encode được tạo lại từ nguồn/checksum và cấu hình normalization, cập nhật provenance tương ứng.

## 1. Chất lượng ROI và non-ROI

VMAF, PSNR/SSIM và ROI PSNR/SSIM của pipeline lấy mẫu mỗi 5 frame. Benchmark giữ cách lấy mẫu riêng và bổ sung một định nghĩa nhất quán để so sánh hai vùng:

- ROI là union của box YOLO gốc trên reference; non-ROI là phần bù của mask ấy. Không dùng padded encoder box vì cần cố định vùng đánh giá giữa các action.
- Chuyển BGR thành grayscale bằng OpenCV; không gọi grayscale này là chính xác plane Y của bitstream YUV. Tên cột `_y` biểu diễn phép đo luma proxy.
- PSNR pooled theo pixel của từng region; ảnh giống hệt dùng convention 100 dB. Region rỗng có NaN, không gán chất lượng hoàn hảo.
- SSIM map dùng phép convolution separable CUDA FP32, có test đối chiếu bản CPU với tolerance 1e−3 cho sai số thứ tự reduction. Công thức dùng Gaussian 11×11 sigma 1.5, C1=(0.01×255)², C2=(0.03×255)². Chỉ lấy tâm có cửa sổ nằm hoàn toàn trong cùng region và trong frame, để không đưa chi tiết ROI sang score non-ROI. Vùng quá nhỏ không có cửa sổ hợp lệ có NaN; raw sample lưu số pixel/tâm để kiểm coverage.
- Aggregate lấy mean metric sampled frame trong GOP. Giữ ROI metric gốc BGR crop thành cột riêng, không thay hoặc trộn với union grayscale metric bổ sung.

`roi_vs_nonroi_paired.parquet` ghép cùng video/GOP/target bitrate, so ΔQP −3/−6/−9 với ΔQP 0. Bảng cho biết được/mất chất lượng của cả hai vùng và chênh bitrate thực. Cùng target không chắc cùng bitrate thực, nên xem thêm rate–distortion.

## 2. Hiệu quả bitrate

`rate_distortion.csv` chứa mean bitrate thực và mean quality của từng video/action. Đường cong riêng cho từng delta-QP, không gộp cảnh quay thành một nguồn giả.

`matched_quality_savings.csv` so từng mức ROI delta với delta 0 ở cùng chất lượng bằng nội suy tuyến tính **rate theo quality**. Chỉ dùng điểm Pareto có quality tăng theo rate, chỉ tính miền quality giao nhau và không ngoại suy. Trung bình saving tương đối tại 101 mức quality trong miền đó. Đây là matched-quality saving của quy tắc này, không phải BD-rate.

Metric gồm VMAF, SSIM toàn ảnh, ROI/non-ROI và detector/semantic retention. Front thiếu hai điểm hoặc không có miền giao nhau ghi status rõ ràng. Không ép thành saving=0. Có thể xem tình huống ROI giúp detection nhưng giảm chất lượng nền.

Bitrate đo payload H.264, chưa gồm transport/container overhead, retransmission hay FEC. Bảng target error phục vụ đánh giá rate-control, không thay thế rate–distortion.

## 3. Bảo toàn thông tin ngữ nghĩa

Chạy lại YOLOv5n cùng weights, classes, thresholds và inference size trên sampled **frame encoded**. Anchor là detections reference đã có, cùng frame index. Matching theo thứ tự confidence của detection encoded, cùng class và IoU≥0.5, một reference chỉ match một lần.

Lưu raw counts và tính micro precision, recall, F1 từ tổng matched/reference/predicted; không trung bình tỷ lệ frame khi denominator bằng 0. Reference không có đối tượng → recall không xác định. Predicted không có đối tượng → precision không xác định; nếu reference vẫn có đối tượng thì recall/F1 bằng 0.

Các metric bổ sung:

- Matched mean IoU: localization drift của đối tượng giữ lại.
- Matched confidence delta/ratio: confidence thay đổi trong các đối tượng match; đọc cùng recall vì chỉ nhìn survivor có bias.
- Semantic retention: semantic mass reference được giữ lại qua match, chia tổng mass reference. Mass của đối tượng = class weight × reference confidence × sqrt(reference area fraction). Không cộng mass từ detection mới để che đối tượng bị mất.
- Semantic score absolute error: chênh `1−exp(−mass)` của detection reference và encoded; metric này có thể nhỏ dù các đối tượng khác nhau, nên không dùng một mình.
- `semantic_by_class.csv`: breakdown video/action/class, counts cùng recall/precision/F1.

Đây là **detector agreement/pseudo-label preservation**, không phải mAP, human ground-truth recall hoặc chứng minh nhận biết mọi vật thể đúng. Detector có thể sai ở reference; detection encoded mới có thể là đối tượng thật. Muốn task accuracy phải bổ sung nhãn độc lập.

## 4. Thích ứng bandwidth

Mọi policy dùng cùng episode và warm-up 4 GOP action 4. Test FCC/Norway giữ nguyên tập episode của `GridDataset.evaluation_specs()`, có thể so với rollout DDQN cũ.

Strategy:

- DDQN selected: checkpoint tốt nhất theo validation, train lại với reward hiện tại.
- DDQN fixed ROI: cùng checkpoint, chọn bitrate bằng Q trong 8 action khóa ROI ΔQP −6; không train riêng.
- CBR mô phỏng: 32 cấu hình target bitrate/ROI cố định, chọn một trên validation. Thêm filler ảo tới target nếu payload thấp hơn; overshoot giữ bitrate thực.
- VBR mô phỏng: target theo semantic `base × (1 + gain × (2S − 1))`, chọn mức bitrate grid gần nhất. Base có 8 mức, gain 0.5/1.0, ROI mode 0/1/2/3/semantic: 80 candidate chọn bằng validation.
- Optimal offline (oracle DP): biết toàn bộ bandwidth và outcome grid; tối đa tổng reward gồm switching penalty. Đây là trần reward trong surrogate.

CBR/VBR không dùng bandwidth để chọn action. Semantic ROI index là tổng điều kiện S≥0.25, S≥0.5, S≥0.75. Hai baseline dùng encode grid hiện có, không phải encoder CBR/constant-quality VBR chạy liên tục.

Reward hiện tại: `0.45*VMAF/100 + 0.25*ROI_SSIM - 1.0*violation - 0.05*switch`. Với `e=max(0,R/max(BW,0.05)-1)`, violation = `min(e nếu e≤0.5, còn lại e²+0.25, 4)`. Switch = `0.5*(|Δbitrate_index|/7 + |ΔROI_index|/3)`. Không phạt backlog/deadline.

Các metric: reward/GOP, return, full/region quality, semantic counts/rates, actual bitrate, delivered throughput, capacity utilization/unused capacity của queue, violation rate, excess Mbit, switch rate và violation ở GOP ngay sau bandwidth giảm hơn 50%. Các metric queue được gắn tên rõ là mô phỏng.

Stress suite cho mỗi video: step từ 8→0.6→4 Mbps, outage tại khoảng giữa rồi hồi phục 3 Mbps, dao động 0.5/6 Mbps mỗi 3 giây. Đây là chẩn đoán trên trace synthetic định nghĩa trước, không sử dụng để chọn baseline hoặc checkpoint.

## 5. Mô hình hàng đợi và giới hạn latency

Queue FIFO, infinite buffer, arrival nguyên payload GOP tại đầu giây, service đều trong giây theo BW. Xếp hàng từ GOP 0 nên warm-up backlog được giữ; metric báo từ GOP 4. Mỗi giây:

```text
initial = previous_backlog + payload_mbit
served = min(initial, bandwidth_mbps × 1s)
backlog_end = initial − served
```

Thời điểm hoàn tất từng GOP tính từ cumulative actual service, không bank capacity khi link idle. GOP phải hoàn tất trong 1 giây kể từ arrival để đạt deadline. Chưa hoàn tất cuối episode là right-censored: latency NaN, `uncompleted=True`, deadline miss; không tự giả định capacity sau trace. p95 latency chỉ trên GOP hoàn tất, phải đọc cùng uncompleted rate.

Mô hình không có RTT, loss, socket scheduling, packetization, frame pacing, encode delay hoặc feedback delay; burst tại đầu giây khác encoder sinh packet dần dần. Không dùng số latency mô phỏng như latency thực đo trên hệ thống.

## 6. So sánh ghép cặp và tái lập

`paired_policy_comparisons.csv` tính DDQN selected − từng baseline trên cùng split/video/trace/offset, báo mean difference theo episode và theo video. CI95% bootstrap resample **video cluster**, 2.000 lần, seed 20261004. Đây là mô tả biến thiên của các video trong split; trace windows giữa video vẫn tương quan nên không diễn giải thành chứng minh tổng quát hóa mạng.

`manifest.json` ghi hash input/code, FPS sampling, hardware/software và elapsed time. Sidecar từng shard ghi hash reference/encode gián tiếp trong job hash và hash kết quả. Đổi weights, config, reference, detection hoặc code làm cache tính lại. Raw encode/reference được xác minh hash với manifest nguồn khi benchmark chạy.

Bộ benchmark vẫn có giới hạn của dataset nguồn: video tách độc lập giữa train/test, semantic của GOP hiện tại nhìn toàn GOP, outcome grid từ action cố định không giữ đúng rate-control khi policy switch, chưa có mạng/VCU thật. Kết quả so sánh là trong phạm vi surrogate này.

Sau chạy, `verification.json` ghi các kiểm tra kết quả và test suite. `manifest.elapsed_s` là thời gian lần gọi hoàn tất cuối cùng, có thể đã dùng cache; tổng thời gian đo các shard được ghi riêng trong verification, không phải tổng thời gian wall-clock của toàn phiên làm việc.

Đo CPU + CUDA trên `bdd_114af10f-c3d93b85`: 6 action mới (5 CPU, 1 CUDA), mỗi action 1.200 frame/240 mẫu, hoàn thành trong 210.965 giây, tương đương 102.387 action/giờ, gồm startup worker. CUDA chạy riêng action 0 trong 107.686 giây, VMAF/PSNR/SSIM/ROI trùng hoàn toàn với mẫu CPU đã lưu. Đây là đo một lượt, chưa bao gồm encode, reference cache, benchmark hoặc train.

So sánh cùng 6 action 0–5 của `bdd_16dce9f0-a914a65e`, 1.200 frame/action, stride 5, cùng reference cache và worker fork: CPU 6 × 2 thread đạt 124.030 giây (174.151 action/giờ); CPU 5 × 2 thread + CUDA 1 đạt 122.816 giây (175.873 action/giờ). Sai lệch tốc độ khoảng 1%, chưa chứng minh lợi ích đáng kể của CUDA. Metrics hai lượt đạt parity. CPU được đo trước; chưa tính encode/reference/benchmark và không dùng số đo này để khẳng định tốc độ toàn pipeline.
