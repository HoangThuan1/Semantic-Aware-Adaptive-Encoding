# Kết quả benchmark video/ROI/bandwidth

## Độ phủ

- 56 video; 2314 GOP; 74048 GOP/action (32 action).
- Đo bổ sung 222144 frame encoded, offsets [0, 14, 29] trong mỗi GOP.
- Metric chất lượng gốc lấy mẫu stride 5; non-ROI và detection dùng frame sample nêu trên.
- Reward dùng chung: `{"action_switch": 0.05, "bandwidth_epsilon_mbps": 0.05, "bandwidth_penalty_shape": "linear_shoulder", "bandwidth_utilization": 0.0, "bandwidth_violation": 1.0, "full_vmaf": 0.45, "penalty_cap": 4.0, "roi_ssim": 0.25}`.

## So sánh trên video và trace test FCC/Norway

| Strategy | Reward/GOP | VMAF | ROI SSIM | Non-ROI SSIM Y | Detection recall | Mbps | Vượt BW | Trễ deadline mô phỏng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ddqn_selected | 0.1975 | 54.44 | 0.8392 | 0.8722 | 0.7695 | 1.131 | 31.57% | 51.36% |
| ddqn_fixed_roi_minus6 | 0.1581 | 44.51 | 0.8142 | 0.8390 | 0.7153 | 0.711 | 19.74% | 32.46% |
| cbr_validation_selected | 0.1234 | 39.39 | 0.7397 | 0.8263 | 0.5962 | 0.592 | 19.41% | 30.74% |
| vbr_validation_selected | 0.1643 | 39.39 | 0.7397 | 0.8263 | 0.5962 | 0.523 | 16.56% | 27.32% |
| oracle_reward_dp | 0.2812 | 57.70 | 0.8689 | 0.8783 | 0.7907 | 1.419 | 26.40% | 43.67% |

## Cách diễn giải

- ROI/non-ROI bổ sung dùng union box reference, grayscale OpenCV, SSIM Gaussian 11×11 sigma 1.5. SSIM chỉ lấy tâm cửa sổ nằm hoàn toàn trong vùng; PSNR dùng toàn pixel vùng. Không trộn với ROI SSIM BGR crop của pipeline gốc.
- Detection precision/recall/F1 là mức đồng thuận với detection YOLO trên reference, same-class IoU≥0.5, matching một-một theo confidence. Không phải mAP hoặc accuracy với nhãn con người; extra detection không chắc là false positive thực.
- Semantic retention là tỷ lệ reference semantic mass của đối tượng được match; confidence ratio chỉ tính trên đối tượng match nên có survivor bias. Xem recall cùng confidence.
- Rate-distortion dùng bitrate thực. Matched-quality saving nội suy tuyến tính trên Pareto front trong miền chất lượng giao nhau, không ngoại suy và không gọi là BD-rate. Giá trị dương là tiết kiệm, âm là tốn thêm bit.
- CBR và VBR được chọn chỉ bằng validation return, không bằng test. Oracle DP biết trước toàn trace và outcome, tối đa hóa tổng reward không discount; là trần reward trong grid, không phải thuật toán triển khai.
- CBR mô phỏng giữ target/ROI cố định, thêm filler ảo đến target khi payload thấp hơn; GOP overshoot giữ bitrate thực. Đây không phải encode CBR mới.
- Fixed ROI dùng cùng DDQN selected, chọn Q cao nhất trong 8 action có ROI ΔQP −6; không train model riêng. Warm-up vẫn dùng action 4 như các phương án khác.
- VBR mô phỏng đổi target theo semantic: base×(1+gain×(2S−1)), chọn mức grid gần nhất; không dùng BW và không phải constant-quality encoder.
- Mọi strategy dùng warm-up action 4 trong 4 GOP. Quality/return bỏ warm-up, nhưng backlog của warm-up được giữ khi mô phỏng hàng đợi.
- Quality là chất lượng video đã encode, bao gồm cả GOP trễ hoặc chưa truyền xong; không phải chất lượng thực tế viewer nhận được qua mạng. Delivered throughput/utilization chỉ là service của mô hình queue.
- Queue là FIFO không giới hạn, payload cả GOP đến đầu giây, link phục vụ đều trong giây; không RTT, loss, overhead hay pacing encoder. Deadline 1 giây. Uncompleted là chưa truyền xong khi trace episode kết thúc; latency của chúng bị censor, p95 chỉ trên GOP hoàn tất.
- Bootstrap ghép cặp theo episode rồi gom theo video, 2.000 bootstrap với seed cố định. Trace windows giữa video có thể tương quan; CI mô tả biến thiên nội dung trong dataset này, không chứng minh tổng quát hóa mạng thật.
- Grid encode action cố định còn domain gap với encode liên tục khi chuyển action. Semantic S[k] cần toàn GOP như pipeline gốc. Video train/validation/test được tách độc lập; chưa có kiểm chứng VCU hoặc mạng thật.

## Artifact

- `grid_enriched.parquet`, `frame_samples.parquet`, `semantic_by_class.csv`: metric bổ sung và raw counts.
- `rate_distortion.csv`, `matched_quality_savings.csv`, `roi_vs_nonroi_paired.parquet`: hiệu quả bitrate/ROI.
- `policy_summary.csv`, `policy_episodes.parquet`, `policy_rollouts.parquet`: validation/test/stress và trace quyết định.
- `baseline_selection.json`, `validation_candidates.csv`: bằng chứng chọn baseline trên validation.
- `paired_policy_comparisons.csv`: difference DDQN−baseline và CI theo video.
- `manifest.json`: SHA-256 input/code, tham số, độ phủ, thời gian và môi trường.
- `rd_*.png`, `policy_comparison.png`: biểu đồ độc lập để xem/chia sẻ; RD chia tối đa 12 video/trang.
