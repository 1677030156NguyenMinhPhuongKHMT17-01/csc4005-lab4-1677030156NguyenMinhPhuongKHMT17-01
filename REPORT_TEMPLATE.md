# CSC4005 Lab 4 Report – CRNN for UrbanSound8K

## 1. Thông tin sinh viên

- Họ tên: Nguyen Minh Phuong
- Mã sinh viên: 1677030156
- Lớp: KHMT17-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab4-1677030156NguyenMinhPhuongKHMT17-01
- Link W&B project: https://wandb.ai/shootingsaviorstar-discord/csc4005-lab4-urbansound8k-crnn

## 2. Mục tiêu thí nghiệm

Lab 4 tiếp tục với Lab 3 bằng cách nâng cấp từ 1D-CNN lên CRNN (Convolutional Recurrent Neural Network) cho bài toán phân loại âm thanh môi trường UrbanSound8K. Thay vì chỉ dùng 1D-CNN học các mẫu cục bộ trên chuỗi đặc trưng, CRNN bổ sung khả năng học quan hệ thời gian thông qua RNN (GRU/LSTM), giúp mô hình quan sát sự biến đổi của âm thanh theo thời gian. Đầu vào là log-mel spectrogram [B, 1, n_mels, time_steps] - mang lại biểu diễn thời gian-tần số trực quan hơn MFCC. Mô hình được đánh giá trên 4 tiêu chí chính: learning curves, confusion matrix, so sánh với 1D-CNN Lab 3, và phân tích khi nào CRNN tốt hơn 1D-CNN.

## 3. Cấu hình dữ liệu

| Thành phần | Giá trị |
|---|---|
| Dataset | UrbanSound8K |
| Số lớp | 10 |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Feature | log-mel spectrogram |
| Sampling rate | 16 kHz |
| Duration | 4 giây |
| n_mels | 64 |
| n_fft | 1024 |
| hop_length | 512 |

## 4. Cấu hình mô hình

| Thành phần | Baseline CRNN-GRU |
|---|---|
| Model | CRNN (small) |
| CNN channels | (16, 32, 64) |
| RNN type | GRU |
| RNN hidden | 96 |
| Bidirectional | No |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Learning rate | 0.001 (ReduceLROnPlateau) |
| Batch size | 32 |
| Epochs | 25 (early stopping, patience=6) |
| Augmentation | Yes (gain, time shift, noise) |
| Trainable params | 71,338 |

## 5. Kết quả huấn luyện

| Run | best_val_acc | test_acc | Macro F1 | Avg epoch time | Params |
|---|---:|---:|---:|---:|---:|
| **logmel_crnn_gru_baseline** | **0.7157** | **0.7360** | **0.7466** | ~206s | 71,338 |

### Chi tiết per-class (Classification Report)

| Lớp | Precision | Recall | F1-score |
|---|---|---|---|
| air_conditioner | 0.8202 | 0.7300 | 0.7725 |
| car_horn | 0.9524 | 0.6061 | 0.7407 |
| children_playing | 0.5294 | 0.7200 | 0.6102 |
| dog_bark | 0.7364 | 0.8100 | 0.7714 |
| drilling | 0.7865 | 0.7000 | 0.7407 |
| engine_idling | 0.7412 | 0.6774 | 0.7079 |
| **gun_shot** | **0.9143** | **1.0000** | **0.9552** |
| **jackhammer** | 0.7395 | **0.9167** | 0.8186 |
| siren | 0.6786 | 0.4578 | 0.5468 |
| street_music | 0.8144 | 0.7900 | 0.8020 |

Baseline CRNN-GRU đạt **test_acc = 0.7360**, cao hơn 1D-CNN log-mel Lab 3 (0.6387) với ít params hơn (71K vs 145K).

## 6. Learning curves

![Learning curves](outputs/logmel_crnn_gru_baseline/curves.png)

Nhận xét:

- **Train loss** giảm đều từ 2.002 -> 0.633 (epoch 25), cho thấy mô hình học rất tốt.
- **Val loss** giảm nhanh ở epoch đầu, dao động nhẹ từ epoch 14 trở đi. Best val_loss = 0.9543 tại epoch 21.
- **Best val_acc = 0.7157** tại epoch 21, mô hình được lưu theo early stopping.
- **Hiệu chỉnh overfitting nhẹ**: train acc đạt 0.7836, val acc dao động 0.63-0.74. Tuy nhiên test_acc (0.7360) tốt hơn val_acc, cho thấy mô hình tổng quát hóa tốt.
- **Learning rate** giảm: 0.001 (epochs 1-18) -> 0.0005 (epochs 19-22) -> 0.00025 (epochs 23-25). ReduceLROnPlateau giúp mô hình ổn định hơn.
- Mô hình chưa trigger early stopping vì val_loss vẫn có xu hướng giảm, cho thấy có thể tăng thêm epochs để cải thiện.

## 7. Confusion matrix

![Confusion matrix](outputs/logmel_crnn_gru_baseline/confusion_matrix.png)

Nhận xét:

- **Lớp phân loại tốt nhất**: **gun_shot** (recall 1.00, f1 0.96) - tiếng súng đạt recall tuyệt đối, dễ phân biệt với bất kỳ âm thanh nào. **jackhammer** (recall 0.92, f1 0.82), **street_music** (f1 0.80), **dog_bark** (recall 0.81, f1 0.77) cũng rất tốt.
- **Lớp bị nhầm lẫn nhiều nhất**: **siren** (recall 0.46) - bị nhầm nhiều sang children_playing (29 của siren bị nhầm thành children_playing). **car_horn** (recall 0.61) - chỉ 33 mẫu trong tập test, và 6 bị nhầm thành street_music.
- **Cặp lớp dễ nhầm**: siren <-> children_playing (29 siren bị nhầm thành children_playing, 16 children_playing bị nhầm thành siren) - hai âm thanh này có tần số và cường độ gần nhau khi có background phức tạp. engine_idling <-> jackhammer (19 engine_idling bị nhầm thành jackhammer) - đều là âm thanh có khua dao động đều.
- **Lợi ích của RNN**: CRNN giảm nhầm lẫn chính tại các lớp có biến đổi theo thời gian rõ ràng (gun_shot, jackhammer, street_music) vì RNN học được các pattern theo thời gian.

## 8. So sánh với Lab 3 1D-CNN

| Tiêu chí | Lab 3: 1D-CNN (log-mel) | Lab 4: CRNN-GRU |
|---|---|---|
| Feature chính | log-mel | log-mel |
| Model | 1D-CNN | CRNN (CNN + GRU) |
| CNN trích đặc trưng cục bộ | Có | Có |
| RNN học quan hệ thời gian | Không | Có |
| **Test accuracy** | 0.6387 | **0.7360** |
| **Best val accuracy** | 0.6890 | **0.7157** |
| **Macro F1** | ~0.63 | **0.7466** |
| Avg epoch time | ~5.6s | ~206s |
| Trainable params | 145,610 | 71,338 |
| Overfitting | Nhẹ | Nhẹ (train 0.78, val 0.72) |

**Phân tích so sánh:**

1. **CRNN-GRU tốt hơn 1D-CNN**: Test accuracy tăng từ 0.6387 lên 0.7360 (+9.73 điểm phần trăm), Macro F1 tăng từ 0.63 lên 0.7466. Đây là cải thiện có ý nghĩa, cho thấy RNN bổ sung khả năng học quan hệ thời gian thực sự giúp phân biệt các âm thanh tốt hơn.

2. **Model nhỏ hơn nhưng hiệu quả hơn**: CRNN chỉ cần 71K params (nhỏ hơn 1D-CNN 145K, giảm 51%) nhưng đạt kết quả cao hơn. CNN trích đặc trưng cục bộ, GRU học quan hệ thời gian - phân chia công việc tốt.

3. **Thời gian train**: CRNN chậm hơn nhiều (~206s/epoch vs ~5.6s/epoch, gấp 37 lần) do RNN tính toán nhiều hơn. Tuy nhiên kết quả tốt hơn nhiều nên trade-off hợp lý.

4. **Lớp được cải thiện nhiều nhất**: gun_shot (recall 0.81 -> 1.00), jackhammer (recall 0.82 -> 0.92), car_horn (f1 0.58 -> 0.74). Các lớp này có biến đổi theo thời gian rõ ràng, CRNN học được.

5. **Lớp vẫn khó**: siren (recall 0.46), children_playing (f1 0.61) vẫn bị nhầm nhiều. Đây là hạn chế của dữ liệu chứ không phải mô hình.

## 9. Kết luận

1. **CRNN-GRU đạt kết quả tốt hơn 1D-CNN**: Test accuracy 0.7360 vs 0.6387 (+9.73%), Macro F1 0.7466 vs 0.63. RNN giúp mô hình học được quan hệ thời gian giữa các pattern âm thanh, đặc biệt với các lớp có cấu trúc thời gian rõ ràng như gun_shot, jackhammer, street_music.

2. **Model nhỏ hơn, hiệu quả hơn**: CRNN chỉ cần 71K params (nhỏ hơn 1D-CNN 145K) nhưng đạt kết quả cao hơn. CNN trích đặc trưng cục bộ, GRU học quan hệ thời gian - phân chia công việc tốt.

3. **Thời gian train nhiều hơn nhưng chấp nhận được**: CRNN cần ~206s/epoch so với ~5.6s của 1D-CNN. Đây là trade-off hợp lý khi cần chất lượng tốt hơn.

4. **Overfitting nhẹ**: Train acc (0.78) và val acc (0.72) chênh nhau 6%, có thể cải thiện bằng thêm regularization hoặc SpecAugment mạnh hơn.

5. **Để cải thiện tiếp**: có thể thử BiLSTM-CRNN (bidirectional), tăng số epochs, thêm SpecAugment, focal loss cho lớp mất cân bằng, hoặc kết hợp với attention mechanism.

## 10. Link minh chứng

- GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab4-1677030156NguyenMinhPhuongKHMT17-01
- W&B project: https://wandb.ai/shootingsaviorstar-discord/csc4005-lab4-urbansound8k-crnn
- W&B baseline run (25 epochs): https://wandb.ai/shootingsaviorstar-discord/csc4005-lab4-urbansound8k-crnn/runs/nkug2spx
