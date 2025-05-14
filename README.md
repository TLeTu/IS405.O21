# ĐỒ ÁN CUỐI KỲ DỮ LIỆU LỚN
## IMPLEMENTATION OF CLASSIFICATION ALGORITHMS FOR PREDICTING CUSTOMER SUBSCRIPTION TO TERM DEPOSITS ON APACHE SPARK

**Lớp**: IS405.021  
**Giảng viên hướng dẫn**: ThS. Nguyễn Hồ Duy Trí  
**TP. Hồ Chí Minh, Năm 2024**

---

## TỔNG QUAN ĐỀ TÀI

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

#### Phân tích dữ liệu

##### Giá trị lớn nhất, giá trị nhỏ nhất:

- **age (tuổi)**  
  - Giá trị lớn nhất (max_age): 95  
  - Giá trị nhỏ nhất (min_age): 18  
  - **Kết luận**: Độ tuổi của các khách hàng trong tập dữ liệu dao động từ 18 đến 95 tuổi, cho thấy tập dữ liệu bao gồm cả người trẻ tuổi và người cao tuổi.

- **balance (số dư tài khoản)**  
  - Giá trị lớn nhất (max_balance): 102127  
  - Giá trị nhỏ nhất (min_balance): -8019  
  - **Kết luận**: Số dư tài khoản dao động từ -8019 đến 102127. Điều này cho thấy có khách hàng bị nợ (số dư âm) và có những khách hàng có số dư rất cao.

- **duration (thời lượng cuộc gọi)**  
  - Giá trị lớn nhất (max_duration): 4918  
  - Giá trị nhỏ nhất (min_duration): 0  
  - **Kết luận**: Thời lượng cuộc gọi dao động từ 0 đến 4918 giây (tương đương hơn 81 phút). Thời lượng bằng 0 có thể cho thấy các cuộc gọi không thành công hoặc bị từ chối ngay lập tức.

- **campaign (số lần liên hệ trong chiến dịch)**  
  - Giá trị lớn nhất (max_campaign): 63  
  - Giá trị nhỏ nhất (min_campaign): 1  
  - **Kết luận**: Số lần liên hệ trong chiến dịch dao động từ 1 đến 63 lần. Điều này cho thấy có khách hàng đã được liên hệ rất nhiều lần trong một chiến dịch.

- **pdays (số ngày kể từ khi khách hàng được liên hệ lần cuối trong chiến dịch trước)**  
  - Giá trị lớn nhất (max_pdays): 536  
  - Giá trị nhỏ nhất (min_pdays): -1  
  - **Kết luận**: Giá trị -1 có thể chỉ ra rằng khách hàng chưa từng được liên hệ trước đó. Số ngày kể từ lần liên hệ trước dao động từ 0 đến 536 ngày đối với những khách hàng đã được liên hệ trước đó.

- **previous (số lần liên hệ với khách hàng trước chiến dịch hiện tại)**  
  - Giá trị lớn nhất (maxPrevious): 275  
  - Giá trị nhỏ nhất (minPrevious): 0  
  - **Kết luận**: Số lần liên hệ trước chiến dịch hiện tại dao động từ 0 đến 275 lần, với nhiều khách hàng chưa từng được liên hệ trước đó.

##### Giá trị phổ biến nhất, hiếm nhất

```python
def get_most_and_least_common(df, col_name):
    counts = df.groupBy(col_name).count().orderBy("count", ascending=False)
    most_common = counts.first()
    least_common = counts.orderBy("count", ascending=True).first()
    return most_common, least_common

rows = []
for attr in attributes:
    most_common, least_common = get_most_and_least_common(df, attr)
    rows.append(Row(statistic=f"most_common_{attr}", value=most_common[attr], count=most_common['count']))
stats_df = spark.createDataFrame(rows)
stats_df.show(truncate=False)
```

**Kết quả thu được:**

| statistic                   | value        | count  |
|-----------------------------|--------------|--------|
| most_common_age             | 32           | 1999   |
| least_common_age            | 94           | 1      |
| most_common_job             | blue-collar  | 9536   |
| least_common_job            | unknown      | 264    |
| most_common_marital         | 2            | 25868  |
| least_common_marital        | 0            | 4965   |
| most_common_education       | secondary    | 22066  |
| least_common_education      | unknown      | 1690   |
| most_common_default         | 0            | 41828  |
| least_common_default        | 1            | 811    |
| most_common_housing         | 1            | 24590  |
| least_common_housing        | 0            | 18049  |
| most_common_loan            | 0            | 35554  |
| least_common_loan           | 1            | 7085   |
| most_common_day             | 20           | 2703   |
| least_common_day            | 1            | 235    |
| most_common_month           | may          | 13532  |
| least_common_month          | dec          | 214    |
| most_common_term_deposit    | 0            | 38678  |
| least_common_term_deposit   | 1            | 3961   |

**Kết luận:**

- **age (tuổi)**  
  - Giá trị phổ biến nhất: 32 (1999 lần xuất hiện)  
  - Giá trị hiếm nhất: 94 (1 lần xuất hiện)  
  - **Nhận xét**: Độ tuổi 32 là độ tuổi phổ biến nhất trong tập dữ liệu, trong khi độ tuổi 94 chỉ xuất hiện duy nhất một lần.

- **job (nghề nghiệp)**  
  - Giá trị phổ biến nhất: blue-collar (9536 lần xuất hiện)  
  - Giá trị hiếm nhất: unknown (264 lần xuất hiện)  
  - **Nhận xét**: Nghề nghiệp blue-collar là phổ biến nhất, còn nghề nghiệp unknown là hiếm nhất trong dữ liệu, điều này có thể chỉ ra rằng phần lớn khách hàng làm các công việc liên quan đến lao động chân tay.

- **marital (tình trạng hôn nhân)**  
  - Giá trị phổ biến nhất: 2 (25868 lần xuất hiện)  
  - Giá trị hiếm nhất: 0 (4965 lần xuất hiện)  
  - **Nhận xét**: Tình trạng hôn nhân 2 (đã kết hôn) là phổ biến nhất, trong khi tình trạng hôn nhân 0 (ly hôn hoặc góa) ít phổ biến hơn.

- **education (trình độ học vấn)**  
  - Giá trị phổ biến nhất: secondary (22066 lần xuất hiện)  
  - Giá trị hiếm nhất: unknown (1690 lần xuất hiện)  
  - **Nhận xét**: Trình độ học vấn secondary là phổ biến nhất trong tập dữ liệu, trong khi unknown là ít phổ biến nhất.

- **default (có thẻ tín dụng)**  
  - Giá trị phổ biến nhất: 0 (41828 lần xuất hiện)  
  - Giá trị hiếm nhất: 1 (811 lần xuất hiện)  
  - **Nhận xét**: Phần lớn khách hàng không có nợ xấu (default = 0), trong khi số lượng khách hàng có nợ xấu (default = 1) là rất ít.

- **housing (vay mua nhà)**  
  - Giá trị phổ biến nhất: 1 (24590 lần xuất hiện)  
  - Giá trị hiếm nhất: 0 (18049 lần xuất hiện)  
  - **Nhận xét**: Số lượng khách hàng có vay mua nhà (housing = 1) nhiều hơn so với những khách hàng không vay mua nhà (housing = 0).

- **loan (vay cá nhân)**  
  - Giá trị phổ biến nhất: 0 (35554 lần xuất hiện)  
  - Giá trị hiếm nhất: 1 (7085 lần xuất hiện)  
  - **Nhận xét**: Phần lớn khách hàng không có vay cá nhân (loan = 0), trong khi số lượng khách hàng có vay cá nhân (loan = 1) ít hơn.

- **day (ngày liên hệ)**  
  - Giá trị phổ biến nhất: 20 (2703 lần xuất hiện)  
  - Giá trị hiếm nhất: 1 (235 lần xuất hiện)  
  - **Nhận xét**: Ngày 20 là ngày phổ biến nhất để liên hệ khách hàng, trong khi ngày 1 là ít phổ biến nhất.

- **month (tháng liên hệ)**  
  - Giá trị phổ biến nhất: may (13532 lần xuất hiện)  
  - Giá trị hiếm nhất: dec (214 lần xuất hiện)  
  - **Nhận xét**: Tháng 5 là tháng phổ biến nhất để liên hệ khách hàng, trong khi tháng 12 là ít phổ biến nhất.

- **term_deposit (quyết định gửi tiết kiệm)**  
  - Giá trị phổ biến nhất: 0 (38678 lần xuất hiện)  
  - Giá trị hiếm nhất: 1 (3961 lần xuất hiện)  
  - **Nhận xét**: Phần lớn khách hàng không chọn gửi tiết kiệm (term_deposit = 0), trong khi số lượng khách hàng chọn gửi tiết kiệm (term_deposit = 1) là ít hơn nhiều.

##### Giá trị trung bình, trung vị

```python
numeric_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
avg_values = df.select(
    [avg(col).alias(f"avg_{col}") for col in numeric_columns]
).collect()[0]
median_values = df.approxQuantile(numeric_columns, [0.5], 0.01)
median_values_dict = {f"median_{col}": median for col, median in zip(numeric_columns, median_values)}
rows = []
for col in numeric_columns:
    rows.append(Row(statistic=f"avg_{col}", value=avg_values[f"avg_{col}"]))
    rows.append(Row(statistic=f"median_{col}", value=median_values_dict[f"median_{col}"][0]))
stats_df = spark.createDataFrame(rows)
stats_df.show(truncate=False)
```

**Kết quả thu được:**

(Tài liệu gốc không cung cấp bảng kết quả cụ thể cho phần này, nhưng đây là đoạn mã dùng để tính giá trị trung bình và trung vị cho các cột số như age, balance, duration, campaign, pdays, previous.)

### Công cụ sử dụng

- **Apache Spark**: Nền tảng xử lý dữ liệu lớn phân tán, được sử dụng để triển khai các thuật toán phân lớp và xử lý dữ liệu.
- **Python**: Ngôn ngữ lập trình chính để viết mã xử lý dữ liệu và triển khai thuật toán.
- **PySpark**: Thư viện Python để tương tác với Apache Spark.
- **Kaggle**: Nguồn cung cấp dữ liệu (dataset: Bank Customer Data in VietNam).

---

## TIỀN XỬ LÝ DỮ LIỆU

### Loại bỏ thuộc tính không cần thiết

- Loại bỏ cột **ID** vì nó chỉ là số thứ tự và không mang giá trị dự đoán.

### Nhận diện và xử lý dữ liệu bị nhiễu

- Sử dụng phương pháp **BoxPlot** để xác định các giá trị ngoại lệ (outliers).

```python
# Code vẽ biểu đồ BoxPlot xác định Outlier
```

**Kết quả**:

- Biểu đồ BoxPlot cho thấy các cột như `balance`, `duration`, `campaign`, `pdays`, `previous` có các giá trị ngoại lệ.
- Xây dựng hàm **QuantileOutlierClipper** để xử lý các giá trị ngoại lệ bằng cách giới hạn chúng trong khoảng từ Q1 - 1.5*IQR đến Q3 + 1.5*IQR.

```python
# Hàm QuantileOutlierClipper
# Áp dụng hàm QuantileOutlierClipper
```

### Nhận diện dữ liệu bị thiếu

- Kiểm tra các giá trị `null` hoặc `unknown` trong các cột như `job`, `education`, `month`.

```python
# Code nhận diện giá trị bị thiếu
```

### Xử lý ý nghĩa dữ liệu

#### Cột 'job' và 'education'

- Chuyển các giá trị `unknown` trong cột `job` và `education` thành giá trị phổ biến nhất (mode) của cột tương ứng.

```python
# Code xử lý 'job' và 'education'
```

#### Cột 'month'

- Chuyển đổi giá trị tháng (chuỗi như 'jan', 'feb',...) thành số nguyên (1 đến 12) để chuẩn hóa dữ liệu.

```python
# Code xử lý cột 'month'
```

### Chuẩn hóa dữ liệu

- Sử dụng **StandardScaler** của PySpark để chuẩn hóa các cột số (`age`, `balance`, `duration`, `campaign`, `pdays`, `previous`) về cùng một thang đo.

```python
# Code chuẩn hóa dữ liệu
```

**Kết quả**:

- Dữ liệu sau khi chuẩn hóa có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1.

### Tách dữ liệu huấn luyện và kiểm thử

- Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%) bằng phương thức `randomSplit` của PySpark.

```python
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
```

**Kết quả**:

- Tập huấn luyện: ~34,080 dòng
- Tập kiểm thử: ~8,520 dòng

---

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

## ĐÁNH GIÁ CÁC THUẬT KHAI THÁC DỮ LIỆU

### Đánh giá về thời gian chạy thuật toán

**Hình 4.1 Đánh giá thời gian chạy**

- Từ kết quả trên, ta thu được thời gian chạy của thuật toán **Support Vector Machine**: **4041s**.
- **Nhận xét**:  
  - SVM thường phức tạp hơn vì nó liên quan đến việc tìm kiếm một siêu phẳng tối ưu để phân tách các lớp, điều này yêu cầu nhiều phép tính lặp lại, đặc biệt khi sử dụng phương pháp SGD trên môi trường phân tán như Apache Spark.

### Đánh giá về độ chính xác tổng quát của thuật toán (Accuracy)

- **Khái niệm**: Accuracy (độ chính xác) là một thước đo hiệu suất của một mô hình phân loại. Accuracy cho biết tỷ lệ phần trăm các dự đoán đúng của mô hình so với tổng số các dự đoán. Nói cách khác, nó là tỷ lệ giữa số lượng dự đoán đúng (bao gồm cả True Positives và True Negatives) trên tổng số các trường hợp được đánh giá.

- **Công thức tính**:  
  $$ \text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Predictions}} $$

**Hình 4.2 Đánh giá Accuracy**

- Từ kết quả trên, ta thu được độ chính xác tổng quát của thuật toán **Support Vector Machine**: khoảng **72.88%**.
- **Nhận xét**:  
  - SVM có độ chính xác thấp hơn so với các thuật toán khác (như KNN đạt 86.38%) trong trường hợp này. Mặc dù SVM thường mạnh mẽ trong việc tìm siêu phẳng tối ưu để phân tách các lớp, có thể có các yếu tố ảnh hưởng như việc lựa chọn tham số không tối ưu, hoặc dữ liệu có thể không phân lớp rõ ràng.

### Đánh giá về độ chính xác của thuật toán (Precision)

- **Khái niệm**: Precision đo lường tỷ lệ các mẫu được dự đoán là dương tính (positive) bởi mô hình mà thực sự là dương tính. Nó cho biết mô hình chính xác đến mức nào khi dự đoán một mẫu là thuộc lớp dương tính.

- **Công thức tính**:  
  $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$

Trong đó:  
- **TP (True Positives)**: Số lượng các mẫu mà mô hình dự đoán là dương tính và thực sự là dương tính.  
- **FP (False Positives)**: Số lượng các mẫu mà mô hình dự đoán là dương tính nhưng thực sự là âm tính.

#### Độ chính xác của thuật toán trên dữ liệu 0

**Hình 4.3 Đánh giá Precision trên dữ liệu 0**

- Từ kết quả trên, ta thu được độ chính xác của thuật toán **Support Vector Machine** trên dữ liệu 0: khoảng **98.74%**.
- **Nhận xét**:  
  - SVM với precision rất cao (98.74%) rất đáng tin cậy trong các dự đoán positive, làm cho nó trở thành lựa chọn tốt trong bài toán khi mà false positives có thể gây ra vấn đề nghiêm trọng.

#### Độ chính xác của thuật toán trên dữ liệu 1

**Hình 4.4 Đánh giá Precision trên dữ liệu 1**

- Từ kết quả trên, ta thu được độ chính xác của thuật toán **Support Vector Machine** trên dữ liệu 1: khoảng **24.58%**.
- **Nhận xét**:  
  - SVM có precision thấp trên tập dữ liệu 1 (24.6%), cho thấy nó không thực sự hiệu quả trong việc phân loại chính xác các điểm positive trên tập dữ liệu này. Sự chênh lệch không đủ lớn để coi là đáng kể, điều này có thể chỉ ra rằng tập dữ liệu này có đặc điểm khó khăn đối với SVM, hoặc cần điều chỉnh thêm các tham số (hyperparameters) hoặc phương pháp tiền xử lý dữ liệu (data preprocessing) để cải thiện hiệu suất.

### Đánh giá về độ nhạy của thuật toán (Recall)

- **Khái niệm**: Recall, còn được gọi là độ nhạy (sensitivity) hoặc tỷ lệ true positive (true positive rate), là một thước đo hiệu suất của một mô hình phân loại, đặc biệt quan trọng trong các bài toán mà việc phát hiện đúng các trường hợp positive là quan trọng. Recall đo lường khả năng của mô hình trong việc nhận diện chính xác tất cả các trường hợp positive. Một recall cao nghĩa là mô hình có khả năng tốt trong việc phát hiện ra các trường hợp positive, giảm thiểu số lượng false negatives (các trường hợp positive bị dự đoán sai là negative).

- **Công thức tính**:  
  $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

Trong đó:  
- **True Positives (TP)**: Số lượng các trường hợp positive được dự đoán đúng bởi mô hình.  
- **False Negatives (FN)**: Số lượng các trường hợp positive bị dự đoán sai là negative bởi mô hình.

#### Độ nhạy của thuật toán trên dữ liệu 0

**Hình 4.5 Đánh giá Recall trên dữ liệu 0**

- **Support Vector Machine**: khoảng **70.97%**.
- **Nhận xét**:  
  - SVM có recall thấp hơn (71.0%), mặc dù độ chính xác của SVM không quá thấp, nhưng nó có xu hướng bỏ sót một số trường hợp `term_deposit = 0`, khiến cho recall của SVM thấp hơn.

#### Độ nhạy của thuật toán trên dữ liệu 1

**Hình 4.6 Đánh giá Recall trên dữ liệu 1**

- Từ kết quả trên, ta thu được độ nhạy của thuật toán **Support Vector Machine** trên dữ liệu 1: khoảng **91.26%**.
- **Nhận xét**:  
  - SVM có recall trên dữ liệu 1 (91.25%) cao hơn rất nhiều so với recall trên dữ liệu 0 (71.0%). Điều này cho thấy SVM hoạt động tốt trong việc nhận diện các trường hợp `term_deposit = 1`, có sự cải thiện đáng kể so với dữ liệu 0.

### 4.5 Nhận xét Support Vector Machine

- Từ kết quả của các độ đo đánh giá, so sánh các thuật toán ở trên, ta có thể suy ra nhận xét về ưu, nhược điểm của thuật toán **SVM** trên tập dữ liệu như sau:

**Ưu điểm**:  
- **Đơn giản và dễ hiểu**: Mô hình SVM tuyến tính sử dụng SGD để tối ưu hóa trọng số là một phương pháp đơn giản và dễ hiểu.  
- **Khả năng song song hóa**: Mô hình này có thể dễ dàng triển khai trên các hệ thống phân tán như Apache Spark, giúp tận dụng tài nguyên của các cụm máy tính để tăng tốc độ huấn luyện.  
- **Tính tổng quát hóa cao**: SVM với biên phân cách tối ưu có khả năng tổng quát hóa tốt trên các tập dữ liệu chưa được nhìn thấy.

**Nhược điểm**:  
- **Thời gian chạy dài**: Quá trình huấn luyện SVM với phương pháp SGD trên tập dữ liệu lớn có thể mất nhiều thời gian, đặc biệt khi số lượng iteration lớn.  
- **Tiêu tốn tài nguyên**: Mặc dù sử dụng SGD giúp giảm thiểu bộ nhớ cần thiết so với các phương pháp tối ưu hóa khác, nhưng với tập dữ liệu quá lớn, yêu cầu về bộ nhớ và tài nguyên tính toán vẫn cao.  
- **Độ chính xác chưa cao**: Độ chính xác của mô hình SVM trên tập dữ liệu kiểm tra chỉ đạt khoảng **73%**, điều này cho thấy mô hình có thể chưa đủ tốt cho những bài toán yêu cầu độ chính xác cao.  
- **Nhạy cảm với tham số**: Hiệu suất của mô hình phụ thuộc vào các tham số như learning rate (alpha) và regularization strength, yêu cầu phải tinh chỉnh cẩn thận để đạt kết quả tốt nhất.  
- **Cập nhật chậm khi gradient nhỏ**: Khi gradient rất nhỏ, quá trình cập nhật trọng số trở nên chậm.
