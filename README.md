# ĐỒ ÁN CUỐI KỲ DỮ LIỆU LỚN
## IMPLEMENTATION OF CLASSIFICATION ALGORITHMS FOR PREDICTING CUSTOMER SUBSCRIPTION TO TERM DEPOSITS ON APACHE SPARK

**Lớp**: IS405.021  
**Giảng viên hướng dẫn**: ThS. Nguyễn Hồ Duy Trí  
**TP. Hồ Chí Minh, Năm 2024**

---
### Mô tả bài toán

Bài toán tập trung vào việc dự đoán khả năng khách hàng đăng ký gửi tiền có kỳ hạn dựa trên dữ liệu từ các chiến dịch tiếp thị trực tiếp (qua cuộc gọi điện thoại) của một tổ chức ngân hàng Việt Nam sử dụng thuật toán phân lớp Support Vector Machine được triển khai trên Apache Spark để xử lý dữ liệu lớn và đưa ra dự đoán chính xác.

### Mô tả dữ liệu

#### Mô tả thành phần dữ liệu
- **Tên dataset**: Bank Customer Data in VietNam
- **Nguồn dữ liệu**: https://www.kaggle.com/datasets/tomculihiddleston/bank-customerdata-in-vietnam/data
- **Mô tả dữ liệu**: Dữ liệu có liên quan đến các chiến dịch tiếp thị trực tiếp (cuộc gọi điện thoại) của một tổ chức ngân hàng Việt Nam. Mục tiêu là dự đoán xem khách hàng có đăng ký gửi tiền có kỳ hạn hay không.

Dữ liệu gồm 42600 dòng và 16 cột dữ liệu, trong đó:

| STT | Tên thuộc tính | Kiểu dữ liệu | Ý nghĩa |
|-----|----------------|--------------|---------|
| 1   | ID             | int          | Số thứ tự của khách hàng đã tư vấn của chiến dịch |
| 2   | age            | int          | Độ tuổi của khách hàng |
| 3   | job            | string       | Mô tả công việc của khách hàng |
| 4   | marital        | int          | Mô tả tình trạng hôn nhân của khách hàng. 0 (ly hôn hoặc góa), 1 (độc thân), 2 (đã kết hôn cùng như có gia đình) |
| 5   | education      | string       | Mô tả trình độ học vấn. "primary" (học hết tiểu học), "secondary" (học hết cấp hai), "tertiary" (từ cấp ba trở lên: cấp ba, đại học, cao học), "unknown" (không xác định) |
| 6   | default        | binary       | Mô tả khách hàng có sử dụng thẻ tín dụng mặc định hay không. 0 (không), 1 (có) |
| 7   | housing        | binary       | Mô tả khách hàng có khoản vay nhà hay không. 0 (không), 1 (có) |
| 8   | loan           | binary       | Mô tả khách hàng có khoản vay cá nhân hay không. 0 (không), 1 (có) |
| 9   | day            | int          | Ngày liên hệ cuối cùng với khách hàng ở chiến dịch lần này |
| 10  | month          | string       | Tháng của ngày liên hệ cuối cùng với khách hàng ở chiến dịch lần này |
| 11  | duration       | int          | Tổng số thời gian liên lạc với khách hàng ở chiến dịch lần này (tính bằng giây) |
| 12  | campaign       | int          | Số lần liên hệ với khách hàng ở chiến dịch này |
| 13  | pdays          | int          | Số ngày trôi qua kể từ lần liên hệ cuối cùng với khách hàng này ở chiến dịch lần trước (-1 tức là khách hàng này chưa được liên hệ trước đó) |
| 14  | previous       | int          | Số lần liên hệ được thực hiện trước chiến dịch này |
| 15  | term_deposit   | binary       | Khách hàng có đăng ký gửi tiền kỳ hạn không? 0 (không), 1 (có) |

### Giải thuật Support Vector Machine
#### Khái niệm
Nguyên lý hoạt động của Support Vector Machine:

- SVM là một thuật toán máy học thường được sử dụng cho các bài toán phân lớp nhị phân.
- SVM hoạt động bằng cách tìm một siêu phẳng (hyperplane) với biên lớn nhất phân chia dữ liệu thành các lớp, siêu phẳng được tính theo công thức: \( w \cdot x - b = 0 \), trong đó
  - \( w, b \) là các trọng số
  - \( x \) là các thuộc tính feature của bộ dữ liệu
- SVM sẽ có công thức như sau:

\[
f(x) = \operatorname{sign}(w \cdot x + b)
\]

Phương pháp Stochastic Gradient Descent:

- Để tính toán được giá trị tối ưu của các trọng số, chúng ta sẽ tính bằng cách giảm tối thiểu giá trị của hàm cost:

\[
J(w) = \frac{1}{2}\|w\|^2 + C \left[ \frac{1}{N} \sum_i^n \max \left(0, 1 - y_i \cdot \left(w \cdot x_i + b\right)\right) \right]
\]

- Chúng ta có thể gộp \( w \) và \( b \) lại thành một bằng cách đưa \( b \) vào vector \( W \) như sau:

\[
f(x_i) = w^{\sim} \cdot x_i^{\sim} + b = W \cdot x_i,
\]

với,

\[
W = (w^{\sim}, b), \quad x_i = (x_i^{\sim}, 1)
\]

- Sau đó thì hàm cost sẽ trở thành:

\[
J(w) = \frac{1}{2}\|w\|^2 + C \left[ \frac{1}{N} \sum_i^n \max \left(0, 1 - y_i \cdot \left(W \cdot x_i\right)\right) \right]
\]

- Dựa vào hàm cost, chúng ta tính đạo hàm riêng (gradient) theo \( W \):

\[
\nabla_W J(w) = \frac{1}{N} \sum_i^n \left\{
\begin{array}{l}
W \text{ if } \max \left(0, 1 - y_i \cdot \left(W \cdot x_i\right)\right) == 0 \\
W - C y_i x_i \quad \text{ otherwise }
\end{array}
\right.
\]

- Và để giảm tối thiểu giá trị của hàm cost thì chúng ta sử dụng phương pháp SGD (Stochastic Gradient Descent):

  - **Bước 1**: Tính đạo hàm riêng của hàm cost.
  - **Bước 2**: Di chuyển trọng số hướng ngược bằng công thức \( W = W - \text{gradient} \).
  - **Bước 3**: Lặp lại đến khi ta tìm được \( W \) sao cho \( J(W) \) đạt nhỏ nhất.
  - Đạo hàm riêng là hướng tăng nhanh nhất của hàm số, vì vậy đi ngược lại sẽ giảm tối thiểu hàm \( J(W) \).

- Lý do phải giảm tối thiểu hàm cost là vì về cơ bản hàm cost là một thước đo mức độ kém hiệu quả của mô hình trong việc đạt được mục tiêu. Nếu nhìn vào \( J(w) \), để tìm giá trị nhỏ nhất của nó, chúng ta phải:
  - Giảm tối thiểu \( ||w||^2 \), nghĩa là tối đa \( \frac{2}{||w||} \) (biên của siêu phẳng).
  - Giảm tối thiểu tổng của \( \max \left(0, 1 - y_i \cdot \left(W \cdot x_i\right)\right) \), nghĩa là tối thiểu việc phân lớp sai.

