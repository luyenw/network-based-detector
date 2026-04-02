# Định nghĩa trọng số đo lường cho bài toán định vị UE

## 1. Trọng số RSRQ

\[
w_i^{(\mathrm{RSRQ})}
= \exp\!\left(
- \frac{( \mathrm{RSRQ}_{\mathrm{ref}} - \mathrm{RSRQ}_i )^2}
{\sigma_{\mathrm{RSRQ}}^2}
\right)
\]

**Giải thích:**

- RSRQ phản ánh chất lượng tín hiệu trên mỗi Resource Block (RB),
  chịu ảnh hưởng bởi nhiễu giao thoa liên cell, tải mạng và phân bổ PRB.
- Sai số RSRQ trong báo cáo UE thường được mô hình hóa xấp xỉ Gaussian.
- \(\mathrm{RSRQ}_{\mathrm{ref}}\): giá trị tham chiếu của liên kết đáng tin cậy.
- \(\sigma_{\mathrm{RSRQ}}\): phương sai cho phép của sai lệch RSRQ,
  thể hiện mức độ chấp nhận dao động quanh giá trị tham chiếu.

---

## 2. Trọng số SNR

\[
w_i^{(\mathrm{SNR})}
= \frac{1}{1 + e^{-(\mathrm{SNR}_i - \theta)}}
\]

**Giải thích:**

- SNR ảnh hưởng tới xác suất lỗi bit và độ chính xác ước lượng RSRP/RSRQ.
- Sau một ngưỡng nhất định, tăng SNR không còn cải thiện độ chính xác.
- Hàm sigmoid được dùng để:
  - Giảm ảnh hưởng của phép đo SNR thấp
  - Không phóng đại quá mức các phép đo SNR rất cao
- \(\theta\): ngưỡng SNR hiệu quả.

---

## 3. Trọng số Timing Advance (TA)

\[
w_i^{(\mathrm{TA})}
= \exp\!\left(
- \frac{( d_i^{\circ}(\mathrm{RSRP}) - d_i^{(\mathrm{TA})} )^2}
{\Delta^2}
\right)
\]

**Giải thích:**

- TA cung cấp ước lượng khoảng cách hình học.
- Nếu khoảng cách suy ra từ TA và RSRP phù hợp → phép đo TA đáng tin.
- Nếu chênh lệch lớn → TA có thể bị nhiễu hoặc lệch cell.
- \(\Delta\): ngưỡng sai số chấp nhận được giữa hai mô hình khoảng cách,
  đóng vai trò như độ lệch chuẩn.

---

## 4. Trọng số ổn định theo thời gian (Variance)

\[
w_i^{(\mathrm{var})}
= \exp\!\left(
- \frac{\mathrm{Var}\!\left(\mathrm{RSRP}_i(t)\right)}
{\sigma_{\mathrm{var}}^2}
\right)
\]

**Giải thích:**

- RSRP biến thiên mạnh cho thấy:
  - Fading nhanh
  - Shadowing
  - UE di chuyển nhanh
- Các phép đo như vậy không phù hợp cho định vị chính xác.
- \(\sigma_{\mathrm{var}}\): ngưỡng dao động RSRP chấp nhận được,
  được ước lượng từ log đo của UE đứng yên.

---

## 5. Trọng số tổng hợp cho RSRP

\[
W_i
= w_i^{(\mathrm{RSRQ})}
\times w_i^{(\mathrm{SNR})}
\times w_i^{(\mathrm{var})}
\]

**Ý nghĩa:**

- Mỗi thành phần phản ánh một xác suất độc lập có điều kiện
  của chất lượng phép đo.
- Dạng tích tương đương với tổng log-likelihood trong mô hình xác suất.
- Nếu một yếu tố chất lượng kém, toàn bộ phép đo sẽ bị giảm trọng số.

---

## 6. Hàm mục tiêu định vị UE

\[
R(\mathbf{u})
=
\sum_i
W_i
\left(
\lVert \mathbf{u} - \mathbf{b}_i \rVert
-
d_i^{\circ}(\mathrm{RSRP}_i)
\right)^2
+
\sum_i
w_i^{(\mathrm{TA})}
\left(
\lVert \mathbf{u} - \mathbf{b}_i \rVert
-
d_i^{(\mathrm{TA})}
\right)^2
\]

**Trong đó:**

- \(\mathbf{u}\): vị trí UE cần ước lượng
- \(\mathbf{b}_i\): vị trí cell \(i\)
- \(d_i^{\circ}(\mathrm{RSRP}_i)\): khoảng cách suy ra từ RSRP
- \(d_i^{(\mathrm{TA})}\): khoảng cách suy ra từ Timing Advance

---

## 7. Diễn giải tổng quát

- \(W_i\): đánh giá độ tin cậy của phép đo RSRP từ cell \(i\)
- \(w_i^{(\mathrm{TA})}\): đánh giá độ tin cậy của ràng buộc hình học từ TA
- Bài toán trở thành **Weighted Least Squares phi tuyến**
  với trọng số thích nghi theo chất lượng kênh và hành vi UE.