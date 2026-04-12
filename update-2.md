# Cải tiến Thuật toán Phát hiện: Phạt Công suất Tuyệt đối (Absolute Power Penalty)

## 1. Hạn chế của phương pháp Chuẩn hóa cũ

Trong thuật toán nguyên bản, để loại bỏ sự phụ thuộc vào độ lợi đường truyền tuyệt đối (absolute path gain) chưa biết, vector RSRP thu được từ các trạm được chuẩn hóa về dạng zero-mean, unit-variance:

$$
\tilde{r}_{u,c} = \frac{r_{u,c} - \mu_u}{\sigma_u}
$$

Chỉ số dị thường (Anomaly Score) của một trạm được tính dựa trên sự bất đồng nhất về mặt hình học sau khi chuẩn hóa:

$$
e_c = \left| \tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u}^*) \right|
$$

**Vấn đề:** 
Khi Trạm giả (FBS) được đặt rất gần hoặc trùng vị trí với Trạm thật (LBS) (khoảng cách $\approx 0$ m hoặc $100$ m) và phát ở công suất cực lớn (ví dụ $60$ dBm, $80$ dBm so với $46$ dBm mặc định), giá trị RSRP $r_{u,f}$ thu được sẽ vọt lên rất cao. Tuy nhiên, sự đột biến này kéo theo giá trị trung bình $\mu_u$ tăng mạnh và dãn rộng độ lệch chuẩn $\sigma_u$. 
Kết quả là sau bước chuẩn hóa, sai số $\tilde{r}_{u,c}$ bị thu hẹp lại, làm "che giấu" đi sự bất thường về công suất. Do FBS ở gần LBS, sai số hình học không đủ lớn để vượt ngưỡng phát hiện.

---

## 2. Đề xuất Cải tiến: Bổ sung Absolute Power Penalty

Để khắc phục điểm mù này, chúng ta kết hợp thêm một hình phạt (penalty) dựa trên **công suất vật lý tuyệt đối**. Nguyên lý là: sóng vô tuyến không thể mạnh hơn mức công suất tối đa lý thuyết phát ra từ trạm thật ở cùng vị trí khoảng cách.

Tại vị trí ước lượng $\mathbf{u}^*$, mức RSRP tối đa theo lý thuyết mà UE có thể nhận được từ trạm thật được tính bằng:

$$
\hat{r}_{u,c}^* = P_c - \text{PL}\bigl(\|\mathbf{u}^* - \mathbf{p}_c\|_2\bigr)
$$

Chúng ta tính phần công suất thu thực tế dư thừa so với mức lý thuyết (chỉ xét khi mức thu thực tế lớn hơn lý thuyết):

$$
P_{\text{excess}} = \max\Bigl(0,\; r_{u,c} - \hat{r}_{u,c}^*\Bigr)
$$

Để đồng bộ với sai số hình học đã chuẩn hóa, giá trị dư thừa này được chia cho một hằng số chuẩn hóa (ví dụ $10.0$ dB, tương đương với một độ lệch chuẩn điển hình của Shadow Fading) để tạo thành điểm phạt:

$$
\text{Penalty}_c = \frac{P_{\text{excess}}}{10.0}
$$

Cuối cùng, hàm tính lỗi cho từng cell (Per-cell error) được cập nhật thành:

$$
e_c = \left| \tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u}^*) \right| + \text{Penalty}_c
$$

## 3. Ý nghĩa

Với cập nhật này, thuật toán tận dụng được cả hai khía cạnh:
1. **Bất đồng nhất hình học (Geometric Inconsistency):** Phát hiện các FBS ở khoảng cách xa (nhờ thành phần thứ nhất $\left| \tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u}^*) \right|$).
2. **Bất thường về năng lượng (Absolute Power Anomaly):** Bắt quả tang các FBS phát công suất "đè" (overpower) LBS ở khoảng cách rất gần (nhờ thành phần thứ hai $\text{Penalty}_c$).