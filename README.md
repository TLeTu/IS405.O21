# ĐỒ ÁN CUỐI KỲ DỮ LIỆU LỚN
## IMPLEMENTATION OF CLASSIFICATION ALGORITHMS FOR PREDICTING CUSTOMER SUBSCRIPTION TO TERM DEPOSITS ON APACHE SPARK

**Lớp**: IS405.021  
**Giảng viên hướng dẫn**: ThS. Nguyễn Hồ Duy Trí  
**TP. Hồ Chí Minh, Năm 2024**

## ÁP DỤNG GIẢI THUẬT KHAI THÁC DỮ LIỆU

### Giải thuật Support Vector Machine

#### Khái niệm

**Nguyên lý hoạt động của Support Vector Machine:**

- SVM là một thuật toán máy học thường được sử dụng cho các bài toán phân lớp nhị phân.
- SVM hoạt động bằng cách tìm một siêu phẳng (hyperplane) với biên lớn nhất phân chia dữ liệu thành các lớp, siêu phẳng được tính theo công thức:  
  $$ w \cdot x - b = 0 $$  
  trong đó:  
  - \( w, b \): các trọng số  
  - \( x \): các thuộc tính feature của bộ dữ liệu  
- SVM sẽ có công thức như sau:  
  $$ f(x) = \text{sign}(w \cdot x + b) $$

**Phương pháp Stochastic Gradient Descent (SGD):**

- Để tính toán được giá trị tối ưu của các trọng số, chúng ta sẽ tìm cách giảm thiểu giá trị của hàm cost:  
  $$ J(w) = \frac{1}{2}\|w\|^2 + C \left[ \frac{1}{N} \sum_{i}^n \max \left(0, 1 - y_i \cdot (w \cdot x_i + b) \right) \right] $$

- Chúng ta có thể gộp \( w \) và \( b \) lại thành một bằng cách đưa \( b \) vào vector \( W \) như sau:  
  $$ f(x_i) = w^\sim \cdot x_i^\sim + b = W \cdot x_i, $$  
  với:  
  $$ W = (w^\sim, b), \quad x_i = (x_i^\sim, 1) $$

- Sau đó, hàm cost sẽ trở thành:  
  $$ J(w) = \frac{1}{2}\|w\|^2 + C \left[ \frac{1}{N} \sum_i^n \max \left(0, 1 - y_i \cdot (W \cdot x_i) \right) \right] $$

- Dựa vào hàm cost, chúng ta tính đạo hàm riêng (gradient) theo \( W \):  
  $$ \nabla_W J(w) = \frac{1}{N} \sum_i^n \begin{cases} 
  W & \text{if } \max \left(0, 1 - y_i \cdot (W \cdot x_i) \right) == 0 \\ 
  W - C y_i x_i & \text{otherwise} 
  \end{cases} $$

- Để giảm tối thiểu giá trị của hàm cost, chúng ta sử dụng phương pháp **SGD (Stochastic Gradient Descent)**:  
  - **Bước 1**: Tính đạo hàm riêng của hàm cost.  
  - **Bước 2**: Di chuyển trọng số hướng ngược gradient bằng công thức:  
    $$ W = W - \text{gradient} $$  
  - **Bước 3**: Lặp lại đến khi tìm được \( W \) sao cho \( J(W) \) đạt nhỏ nhất.  

- Đạo hàm riêng là hướng tăng nhanh nhất của hàm số, vì vậy đi ngược lại sẽ giảm tối thiểu hàm \( J(W) \).

- Lý do phải **phải giảm tối thiểu hàm cost** là vì về cơ bản hàm cost là một thước đo mức độ kém hiệu quả của mô hình trong việc đạt được mục tiêu. Để tìm giá trị nhỏ nhất của \( J(w) \), chúng ta phải:  
  - Giảm tối thiểu \( \|w\|^2 \), nghĩa là tối đa hóa \( \frac{2}{\|w\|} \) (biên của siêu phẳng).  
  - Giảm tối thiểu tổng của \( \max \left(0, 1 - y_i \cdot (W \cdot x_i) \right) \), nghĩa là tối thiểu việc phân lớp sai.

#### Song song hóa giải thuật dựa trên MapReduce

**Ý tưởng:**

- Đầu tiên, khởi tạo một mảng trọng số \( W \) (global \( W \)) với giá trị toàn bộ là 0, sau đó tiến hành training. Trong quá trình training:  
  - Tập dữ liệu train sẽ được chia nhỏ ra thành từng phần (data chunk).  
  - Các **mapper** sẽ tiến hành tính toán gradient của hàm cost dựa trên data chunk nhận được và global \( W \).  
  - Các giá trị gradient sau đó sẽ được gửi đến **reducer** và tiến hành quá trình reduce.  
  - Cuối cùng, dựa trên output của reducer, cập nhật global \( W \).  
  - Tiến hành lặp lại quá trình training dựa vào global \( W \) đã cập nhật cho đến khi thu được trọng số \( W \) mong muốn.

**Cài đặt:**

- Cập nhật lại bộ dữ liệu train/test, thêm vào \( X_train \) và \( X_test \) một cột là `intercept` với toàn bộ giá trị là 1 (dựa theo cách gộp \( w \) và \( b \) ở phần khái niệm):  

```python
train_data, test_data = encoded_df.randomSplit([0.8, 0.2], seed=42)
X_train = train_data.drop('term_deposit')
X_train = X_train.withColumn('intercept', lit(1))
y_train = train_data.select('term_deposit')
X_test = test_data.drop('term_deposit')
X_test = X_test.withColumn('intercept', lit(1))
y_test = test_data.select('term_deposit')
```

- Khởi tạo các RDD, với \( y \) sẽ chỉ chứa 1 và -1:  

```python
X_train_rdd = X_train.rdd.map(lambda row: np.array(row))
y_train_rdd = y_train.rdd.map(lambda row: row['term_deposit'])
y_train_rdd = y_train_rdd.map(lambda x: 1 if x == 1 else -1)
X_test_rdd = X_test.rdd.map(lambda row: np.array(row))
y_test_rdd = y_test.rdd.map(lambda row: row['term_deposit'])
y_test_rdd = y_test_rdd.map(lambda x: 1 if x == 1 else -1)
train_rdd = X_train_rdd.zip(y_train_rdd)
test_rdd = X_test_rdd.zip(y_test_rdd)
```

- Thiết lập số lượng partition trong `train_rdd`, tương tự số lượng data chunk mà dữ liệu sẽ có:  

```python
num_partitions = 4
train_rdd = train_rdd.repartition(num_partitions)
```

- Xây dựng hàm `svm_calculate_cost_gradient` để tính toán giá trị gradient của hàm cost:  

```python
def svm_calculate_cost_gradient(data_iter, W):
    reg_strength = 10000
    data = list(data_iter)
    if len(data) == 0:
        return np.zeros(len(data[0][0]))
    X_batch = []
    Y_batch = []
    for features, label in data:
        X_batch.append(features)
        Y_batch.append(label)
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(data[0][0]))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw / len(Y_batch)
    return [dw]
```

- **Đầu vào** của hàm tính toán là dữ liệu và một mảng trọng số \( W \).  
- **Bước 1**: Chia dữ liệu vào hai mảng là `X_batch` và `Y_batch`. `X_batch` sẽ chứa các thuộc tính \( x \) và `Y_batch` sẽ chứa thuộc tính \( y \).  
- **Bước 2**: Tính toán `distance` bằng `X_batch`, `Y_batch` và \( W \) dựa theo công thức:  
  $$ 1 - y_i \cdot (W \cdot x_i) $$  
- **Bước 3**: Tạo một mảng 0 là `dw`.  
- **Bước 4**: Tiến hành tính toán gradient dựa trên công thức:  
  $$ \nabla_W J(w) = \frac{1}{N} \sum_i^n \begin{cases} 
  W & \text{if } \max \left(0, 1 - y_i \cdot (W \cdot x_i) \right) == 0 \\ 
  W - C y_i x_i & \text{otherwise} 
  \end{cases} $$  
  - Nếu \( \max(0, \text{distance}) == 0 \), thì lấy \( W \).  
  - Ngược lại, thì lấy \( W - (\text{reg_strength} * Y_batch[ind] * X_batch[ind]) \).  
  - Sau đó, cập nhật mảng `dw`.  
- Sau khi kết thúc vòng lặp, chia `dw` cho `len(Y_batch)` tức \( N \).  
- **Đầu ra** của hàm sẽ là một mảng các giá trị gradient.

- Xây dựng hàm `svm_combine_cost_gradient`, tính tổng hai mảng gradient:  

```python
def svm_combine_cost_gradient(dw1, dw2):
    return (dw1 + dw2)
```

- Xây dựng hàm `svm_sgd`, đây sẽ được xem là hàm train:  
  - **Đầu vào** của hàm sẽ là tập dữ liệu các thuộc tính feature (`X_train`).  
  - **Bước 1**: Khởi tạo một mảng trọng số \( W \) (global \( W \)) ban đầu sẽ có giá trị toàn bộ là 0.  
  - **Bước 2**: Khởi tạo vòng lặp training, số lần lặp sẽ tùy vào người sử dụng, điều kiện dừng sẽ dựa trên sự thay đổi của các giá trị gradient, nếu sự thay đổi bé hơn threshold thì dừng vòng lặp, threshold cũng sẽ tùy vào người sử dụng.  
  - Bên trong vòng lặp sẽ tiến hành tính toán các gradient, hàm `mapPartitions` sẽ lấy các partition (data chunk) bên trong \( X \) và đưa các partition đến các mapper để tính toán song song, hàm Map ở đây sẽ là `svm_calculate_cost_gradient`.  
  - Sau khi tính toán, các output sẽ được reduce bằng hàm `svm_combine_cost_gradient`.  
  - Sau đó, output của reduce sẽ chia cho số lượng partition để cho ra kết quả trung bình các gradient.  
  - Cuối cùng, global \( W \) sẽ được cập nhật dựa trên gradient theo công thức:  
    $$ W = W - \text{gradient} $$  
  - Vòng lặp sẽ tiếp tục với global \( W \) vừa được cập nhật cho đến khi kết thúc.  
  - **Đầu ra** của hàm sẽ là một mảng trọng số \( W \) tối ưu.

- Xây dựng hàm `svm_predict`, đây sẽ là hàm sử dụng để dự đoán:  

```python
def svm_predict(W, X):
    pred = np.dot(X, W)
    result = 1 if pred > 0 else -1
    return result
```

- Hàm được xây dựng dựa trên công thức:  
  $$ f(x_i) = W \cdot x_i $$  
- Kết quả dự đoán > 0 thì sẽ thuộc lớp 1, ngược lại thì sẽ thuộc lớp -1.

- Tiến hành train và predict:  

```python
import time
start_time = time.time()
svm_weights = svm_sgd(train_rdd)
end_time = time.time()
predictions_rdd = test_rdd.map(lambda x: (x[1], svm_predict(svm_weights, x[0])))
```

- Thực hiện đánh giá độ chính xác, thời gian chạy và ma trận nhầm lẫn của mô hình, kết quả thu được:  

```
Accuracy: 0.7287757973733584
Time running: 4041.041754722595s
+-----+----------+-----+
|actual|prediction|count|
+-----+----------+-----+
|  0   |    1     | 2243|
|  0   |    0     | 5484|
|  1   |    0     |  70 |
|  1   |    1     | 731 |
+-----+----------+-----+
```

**Từ kết quả thu được, ta có thể thấy:**

- **Độ chính xác** của mô hình vào khoảng **72.9%**, một giá trị tương đối ổn với một mô hình được vận hành trên dữ liệu lớn.
- **Thời gian chạy** của mô hình lên tới **4041s** (khoảng hơn 1 tiếng).
- **Ma trận nhầm lẫn** tương ứng.

---
