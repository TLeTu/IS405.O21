# ĐỒ ÁN CUỐI KỲ DỮ LIỆU LỚN  
## IMPLEMENTATION OF CLASSIFICATION ALGORITHMS FOR PREDICTING CUSTOMER SUBSCRIPTION TO TERM DEPOSITS ON APACHE SPARK

**Lớp:** IS405.021  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hồ Duy Trí  
**TP. Hồ Chí Minh, Năm 2024**

---

## 1. Mô tả dữ liệu

- **Tên dataset:** Bank Customer Data in VietNam  
- **Nguồn dữ liệu:** [Kaggle Dataset](https://www.kaggle.com/datasets/tomculihiddleston/bank-customer-data-in-vietnam/data)  
- **Mô tả:**  
  Dữ liệu liên quan đến các chiến dịch tiếp thị trực tiếp (cuộc gọi điện thoại) của một ngân hàng Việt Nam.  
  Mục tiêu: Dự đoán khách hàng có đăng ký gửi tiền có kỳ hạn hay không.

![Dataset Overview](https://github.com/user-attachments/assets/8f653d77-7df3-4694-851b-172f442fd23b)

- **Kích thước:** 42,600 dòng × 16 cột.

![Dataset Columns Part 1](https://github.com/user-attachments/assets/8cbca48c-af94-41bb-9973-b9731c23df14)  
![Dataset Columns Part 2](https://github.com/user-attachments/assets/10e3705a-7c8b-4384-ac18-d0ab20f1d241)

---

## 2. Phân tích dữ liệu

### 2.1 Giá trị lớn nhất & nhỏ nhất
![Max Min Table](https://github.com/user-attachments/assets/8fb3c135-e03f-497d-a8ed-1ba3a427815d)  
![Max Min Table 2](https://github.com/user-attachments/assets/f7830be1-007c-49ce-acb9-6af4f39e0f1b)

### 2.2 Nhận xét

- **age (tuổi)**  
  - Min: 18  
  - Max: 95  
  → Dữ liệu bao gồm cả người trẻ và người cao tuổi.

- **balance (số dư tài khoản)**  
  - Min: -8019  
  - Max: 102127  
  → Có khách hàng bị nợ (âm) và khách hàng có số dư rất cao.

- **duration (thời lượng cuộc gọi)**  
  - Min: 0  
  - Max: 4918 giây (~81 phút)  
  → 0 giây có thể là cuộc gọi thất bại hoặc bị từ chối.

- **campaign (số lần liên hệ trong chiến dịch)**  
  - Min: 1  
  - Max: 63  
  → Có khách hàng được liên hệ rất nhiều lần.

- **pdays (số ngày từ lần liên hệ trước)**  
  - Min: -1 (chưa từng liên hệ)  
  - Max: 536  

- **previous (số lần liên hệ trước chiến dịch hiện tại)**  
  - Min: 0  
  - Max: 275  

---

## 3. Support Vector Machine (SVM)

![SVM Result 1](https://github.com/user-attachments/assets/e4aedb4c-7cd6-4835-8eba-b4da58909796)  
![SVM Result 2](https://github.com/user-attachments/assets/e1b3fdf1-04de-4279-877d-28a398e79fd6)

---

## 4. Song song hóa giải thuật dựa trên MapReduce

### 4.1 Ý tưởng
![MapReduce Idea](https://github.com/user-attachments/assets/e5fc65bb-bb42-4e0a-ab27-b7d9ac524cfc)

### 4.2 Cài đặt
![MapReduce Code 1](https://github.com/user-attachments/assets/ac9889c3-44e8-49f3-a5da-04dfe5c12931)  
![MapReduce Code 2](https://github.com/user-attachments/assets/e8a5c5f8-b03f-4906-8d5b-d2ad824958a6)  
![MapReduce Code 3](https://github.com/user-attachments/assets/c9d47bb7-6d07-48e0-83d2-9dba5c844385)  
![MapReduce Code 4](https://github.com/user-attachments/assets/810939fd-c39d-4342-b75c-25632ec21146)  
![MapReduce Code 5](https://github.com/user-attachments/assets/de4f669c-7a64-4119-bde9-2c93395ed91c)

---

## 5. Kết luận
(Thêm phần nhận xét, so sánh kết quả, và hướng phát triển ở đây)

