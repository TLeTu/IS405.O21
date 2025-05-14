# Triển khai thuật toán phân loại để dự đoán việc khách hàng đăng ký gửi tiền có kỳ hạn trên Apache Spark
1/ Mô tả bài toán
Bài toán của đề tài là dự đoán khả năng khách hàng đăng ký gửi tiền có kỳ hạn từ các chiến dịch tiếp thị của một tổ chức ngân hàng Việt Nam. Cụ thể, mục tiêu là áp dụng thuật toán phân lớp sử dụng dữ liệu khách hàng, thông tin từ các chiến dịch iếp thị và các yếu tố liên quan khác để dự đoán khả năng khách hàng sẽ đăng ký gửi tiền có kỳ hạn. Nhằm tối ưu hóa chiến lược tiếp thị, tập trung vào những khách hàng có khả năng cao nhất để tăng doanh số.

2/ Mô tả dữ liệu
- Tên dataset: Bank Customer Data in VietNam
- Nguồn dữ liệu: https://www.kaggle.com/datasets/tomculihiddleston/bank-customer-data-in-vietnam/data
- Mô tả dữ liệu: Dữ liệu có liên quan đến các chiến dịch tiếp thị trực tiếp (cuộc gọi điện thoại) của một tổ chức ngân hàng Việt Nam. Mục tiêu là dự đoán xem khách hàng có đăng ký gửi tiền có kỳ hạn hay không.

![image](https://github.com/user-attachments/assets/52109a97-cd76-4190-803a-9766fa8aec7d)

