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
