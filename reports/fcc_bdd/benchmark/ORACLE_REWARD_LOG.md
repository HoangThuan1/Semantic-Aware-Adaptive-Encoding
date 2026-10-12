# Log thử reward oracle DP — 07–08/10/2026

Oracle được tối ưu lại trên cùng validation/test/stress; không train lại DDQN. Bảng dưới là kết quả test FCC/Norway.

“Oracle penalty cũ” trong so sánh cuối là **lần 2**: penalty bình phương, violation 0,50, switching 0,05; không phải reward train DDQN.

| Lần | Penalty BW | Violation / switching | VMAF | ROI SSIM | Mbps | Vượt BW | Đổi action | Trễ mô phỏng |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 — benchmark gốc | Bình phương | 0,30 / 0 | 63,94 | 0,8986 | 1,782 | 81,34% | 58,68% | 98,28% |
| 2 — tăng trọng số, thêm switching | Bình phương | 0,50 / 0,05 | 62,22 | 0,8911 | 1,668 | 69,35% | 34,74% | 94,85% |
| 3 — phạt mạnh vùng 1×–1,5× | Tuyến tính rồi bình phương | 0,50 / 0,05 | 58,94 | 0,8768 | 1,476 | 36,17% | 36,15% | 57,44% |
| 4 — tăng phạt BW (08/10) | Tuyến tính rồi bình phương | 1,00 / 0,05 | 57,70 | 0,8689 | 1,419 | 26,40% | 38,41% | 43,67% |

```text
e = max(0, R_actual / max(BW, 0.05) - 1)
penalty cũ = min(e², 4)
penalty mới = min(e nếu e <= 0.5, còn lại e² + 0.25, 4)
switch = 0.5 × (|Δbitrate_index|/7 + |ΔROI_index|/3)
reward lần 3 = 0.45×VMAF/100 + 0.25×ROI_SSIM − 0.50×penalty_mới − 0.05×switch
reward lần 4 = 0.45×VMAF/100 + 0.25×ROI_SSIM − 1.00×penalty_mới − 0.05×switch
```

Ở mức vượt 10%: điểm phạt tăng 0,005→0,050; vượt 50%: 0,125→0,250. Hàm nối liên tục tại 1,5×. Không có ROI thì thành phần ROI bằng 0.

Cùng chấm bằng reward lần 3: oracle lần 2 đạt **0,3606 reward/GOP**, oracle lần 3 đạt **0,3783**. Không so trực tiếp reward giữa các objective khác nhau.

21 test pass, gồm DP đối chiếu vét cạn cho hai penalty; kiểm tra 12.565 episode xác nhận oracle lần 3 không kém lần 2 dưới reward mới. Penalty mới giảm vượt BW/trễ nhưng giảm chất lượng; reward vẫn không phạt backlog/deadline. Ở thời điểm thử lần 3, DDQN chưa được train lại.

Đã gộp và xóa output chi tiết của các thí nghiệm reward. Công thức chung: `src/vcu_surrogate/rl_env.py::bandwidth_penalty`, `reward_for`; oracle: `src/vcu_surrogate/benchmark.py::oracle_actions`.

Lần 4: cùng 10.446 episode test / 350.562 GOP sau warm-up, avg BW **1,371272 Mbps**, bitrate **1,419454 Mbps**; mean backlog **4,491390 Mbit**. Chạy đủ validation/test/stress (12.565 episode), 23 test pass gồm DP vét cạn ở violation 1,00. Phạt mềm giảm vượt BW nhưng chưa bảo đảm avg bitrate ≤ avg BW; không so reward khác objective. Chỉ giữ log này, đã xóa runner/container tạm; DDQN chưa được train lại tại thời điểm thử oracle.

## Train và benchmark hoàn tất — 08/10/2026, 02:54 (+07)

Reward lần 4 đã áp dụng cho train và mọi baseline. Seed 13 train đủ **300.000 step**; chọn checkpoint **110.000**, validation mean return **3,898130**. Export NumPy khớp Q-value PyTorch (max error 0). Đủ 1.792 measurement cache được xác minh và dùng lại; validate pass.

Cùng **10.446 episode test / 350.562 GOP**, avg BW **1,371272 Mbps**:

| Phương án | Reward/GOP | VMAF | ROI SSIM | Mbps | Vượt BW | Trễ mô phỏng |
|---|---:|---:|---:|---:|---:|---:|
| DDQN | 0,1975 | 54,44 | 0,8392 | 1,131 | 31,57% | 51,36% |
| DDQN fixed ROI −6 | 0,1581 | 44,51 | 0,8142 | 0,711 | 19,74% | 32,46% |
| CBR mô phỏng | 0,1234 | 39,39 | 0,7397 | 0,592 | 19,41% | 30,74% |
| VBR mô phỏng | 0,1643 | 39,39 | 0,7397 | 0,523 | 16,56% | 27,32% |
| Optimal offline DP | 0,2812 | 57,70 | 0,8689 | 1,419 | 26,40% | 43,67% |

Fixed ROI dùng cùng DDQN, argmax Q trong 8 action có ΔQP −6. CBR/VBR chọn từ 112 ứng viên trên validation: CBR target 0,5 Mbps / ROI −3, có filler ảo; VBR base 0,5 / gain 0,5 / ROI −3 theo semantic, không dùng BW. Đây là mô phỏng trên grid encode, không encode CBR/VBR mới. Oracle biết trước trace và tối ưu tổng reward; penalty mềm vẫn cho phép vượt BW.

Đã kiểm tra đủ 12.565 episode validation/test/stress: oracle return không thấp hơn từng policy; fixed ROI luôn −6 sau warm-up; CBR/VBR payload và coverage đúng. Suite 84 pass / 1 skip; benchmark tests pass sau sửa chạy song song. Đã bỏ heuristic; xóa runner/test file phụ, bản sao lưu và tài liệu dài. Bảng đầy đủ: [REPORT.md](REPORT.md).
