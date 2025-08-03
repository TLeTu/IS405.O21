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
![Max Min Code](https://github.com/user-attachments/assets/8fb3c135-e03f-497d-a8ed-1ba3a427815d)  
![Max Min Result](https://github.com/user-attachments/assets/f7830be1-007c-49ce-acb9-6af4f39e0f1b)

- **age (tuổi)**  
  - Min: 18  
  - Max: 95  
  → Độ tuổi của các khách hàng trong tập dữ liệu dao động từ 18 đến 95 tuổi, cho thấy tập dữ liệu bao gồm cả người trẻ tuổi và người cao tuổi.

- **balance (số dư tài khoản)**  
  - Min: -8019  
  - Max: 102127  
  → Số dư tài khoản dao động từ -8019 đến 102127. Điều này cho thấy có khách hàng bị nợ (số dư âm) và có những khách hàng có số dư rất cao.

- **duration (thời lượng cuộc gọi)**  
  - Min: 0  
  - Max: 4918 giây (~81 phút)  
  → Thời lượng cuộc gọi dao động từ 0 đến 4918 giây (tương đương hơn 81 phút). Thời lượng bằng 0 có thể cho thấy các cuộc gọi không thành công hoặc bị từ chối ngay lập tức.

- **campaign (số lần liên hệ trong chiến dịch)**  
  - Min: 1  
  - Max: 63  
  → Số lần liên hệ trong chiến dịch dao động từ 1 đến 63 lần. Điều này cho thấy có khách hàng đã được liên hệ rất nhiều lần trong một chiến dịch.

- **pdays (số ngày từ lần liên hệ trước)**  
  - Min: -1 (chưa từng liên hệ)  
  - Max: 536  
  → Giá trị -1 có thể chỉ ra rằng khách hàng chưa từng được liên hệ trước đó. Số ngày kể từ lần liên hệ trước dao động từ 0 đến 536 ngày đối với những khách hàng đã được liên hệ trước đó.
  
- **previous (số lần liên hệ trước chiến dịch hiện tại)**  
  - Min: 0  
  - Max: 275  
  → Số lần liên hệ trước chiến dịch hiện tại dao động từ 0 đến 275 lần, với nhiều khách hàng chưa từng được liên hệ trước đó.

### 2.2 Giá trị phổ biến nhất, hiếm nhất
![Most Least Code](https://github.com/user-attachments/assets/07e1a1c6-577a-4ea2-b65b-c36c379553c9)  
![Most Least Result](https://github.com/user-attachments/assets/eed07205-bc21-4964-8182-c647ba0c60ee)



---

## 3. Support Vector Machine (SVM)

![SVM 1](https://github.com/user-attachments/assets/e4aedb4c-7cd6-4835-8eba-b4da58909796)  
![SVM 2](https://github.com/user-attachments/assets/e1b3fdf1-04de-4279-877d-28a398e79fd6)

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

