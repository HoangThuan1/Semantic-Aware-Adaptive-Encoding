# Kết quả benchmark video/ROI/bandwidth

## Độ phủ

- 5 video; 294 GOP; 9408 GOP/action (32 action).
- Đo bổ sung 28224 frame encoded, offsets [0, 14, 29] trong mỗi GOP.
- VMAF, full PSNR/SSIM và ROI metric gốc dùng mọi frame; non-ROI và detection dùng frame sample nêu trên.

## So sánh trên trace test Verizon

| Strategy | Reward/GOP | VMAF | ROI SSIM | Non-ROI SSIM Y | Detection recall | Mbps | Vượt BW | Trễ deadline mô phỏng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ddqn_selected | 0.4200 | 73.43 | 0.9231 | 0.9398 | 0.8827 | 3.259 | 21.75% | 30.96% |
| fixed_validation_selected | 0.3657 | 57.32 | 0.9000 | 0.9068 | 0.8476 | 1.488 | 14.36% | 18.24% |
| heuristic_validation_selected | 0.4549 | 79.39 | 0.9496 | 0.9439 | 0.9059 | 4.973 | 31.38% | 44.12% |
| causal_harmonic_roi0 | 0.4358 | 75.94 | 0.8854 | 0.9418 | 0.8368 | 3.968 | 19.57% | 25.90% |
| causal_harmonic_roisemantic | 0.4364 | 75.91 | 0.8936 | 0.9404 | 0.8480 | 3.977 | 19.64% | 26.08% |
| oracle_reward_dp | 0.5065 | 88.26 | 0.9588 | 0.9671 | 0.9194 | 5.575 | 41.42% | 55.77% |

## Cách diễn giải

- ROI/non-ROI bổ sung dùng union box reference, grayscale OpenCV, SSIM Gaussian 11×11 sigma 1.5. SSIM chỉ lấy tâm cửa sổ nằm hoàn toàn trong vùng; PSNR dùng toàn pixel vùng. Không trộn với ROI SSIM BGR crop của pipeline gốc.
- Detection precision/recall/F1 là mức đồng thuận với detection YOLO trên reference, same-class IoU≥0.5, matching một-một theo confidence. Không phải mAP hoặc accuracy với nhãn con người; extra detection không chắc là false positive thực.
- Semantic retention là tỷ lệ reference semantic mass của đối tượng được match; confidence ratio chỉ tính trên đối tượng match nên có survivor bias. Xem recall cùng confidence.
- Rate-distortion dùng bitrate thực. Matched-quality saving nội suy tuyến tính trên Pareto front trong miền chất lượng giao nhau, không ngoại suy và không gọi là BD-rate. Giá trị dương là tiết kiệm, âm là tốn thêm bit.
- Fixed/heuristic tốt nhất được chọn chỉ bằng validation return, không bằng test. Oracle DP biết trước toàn trace và outcome, tối đa hóa tổng reward không discount; là trần reward trong grid, không phải thuật toán triển khai.
- Mọi strategy dùng warm-up action 4 trong 4 GOP. Quality/return bỏ warm-up, nhưng backlog của warm-up được giữ khi mô phỏng hàng đợi.
- Quality là chất lượng video đã encode, bao gồm cả GOP trễ hoặc chưa truyền xong; không phải chất lượng thực tế viewer nhận được qua mạng. Delivered throughput/utilization chỉ là service của mô hình queue.
- Queue là FIFO không giới hạn, payload cả GOP đến đầu giây, link phục vụ đều trong giây; không RTT, loss, overhead hay pacing encoder. Deadline 1 giây. Uncompleted là chưa truyền xong khi trace episode kết thúc; latency của chúng bị censor, p95 chỉ trên GOP hoàn tất.
- Bootstrap ghép cặp theo episode rồi gom theo video, 2.000 bootstrap với seed cố định. Chỉ có 5 video cluster, trace windows giữa video có thể tương quan; CI mô tả biến thiên nội dung trong dataset này, không chứng minh tổng quát hóa LTE.
- Grid encode action cố định còn domain gap với encode liên tục khi chuyển action. Semantic S[k] cần toàn GOP như pipeline gốc. Video train/test dùng chung; chưa có kiểm chứng VCU hoặc mạng thật.

## Artifact

- `grid_enriched.parquet`, `frame_samples.parquet`, `semantic_by_class.csv`: metric bổ sung và raw counts.
- `rate_distortion.csv`, `matched_quality_savings.csv`, `roi_vs_nonroi_paired.parquet`: hiệu quả bitrate/ROI.
- `policy_summary.csv`, `policy_episodes.parquet`, `policy_rollouts.parquet`: validation/test/stress và trace quyết định.
- `baseline_selection.json`, `validation_candidates.csv`: bằng chứng chọn baseline trên validation.
- `paired_policy_comparisons.csv`: difference DDQN−baseline và CI theo video.
- `manifest.json`: SHA-256 input/code, tham số, độ phủ, thời gian và môi trường.
- `rd_*.png`, `policy_comparison.png`: biểu đồ độc lập để xem/chia sẻ.
