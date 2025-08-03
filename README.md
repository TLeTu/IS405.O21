# ĐỒ ÁN CUỐI KỲ DỮ LIỆU LỚN
## IMPLEMENTATION OF CLASSIFICATION ALGORITHMS FOR PREDICTING CUSTOMER SUBSCRIPTION TO TERM DEPOSITS ON APACHE SPARK

**Lớp**: IS405.021  
**Giảng viên hướng dẫn**: ThS. Nguyễn Hồ Duy Trí  
**TP. Hồ Chí Minh, Năm 2024**

### Mô tả dữ liệu

- Tên dataset: Bank Customer Data in VietNam
- Nguồn dữ liệu: https://www.kaggle.com/datasets/tomculihiddleston/bank-customer-data-in-vietnam/data
- Mô tả dữ liệu:  Dữ liệu có liên quan đến các chiến dịch tiếp thị trực tiếp (cuộc gọi điện thoại) của một tổ chức ngân hàng Việt Nam. Mục tiêu là dự đoán xem khách hàng có đăng ký gửi tiền có kỳ hạn hay không.
<img width="1597" height="662" alt="image" src="https://github.com/user-attachments/assets/8f653d77-7df3-4694-851b-172f442fd23b" />

-Dữ liệu gồm 42600 dòng và 16 cột dữ liệu, trong đó:
<img width="799" height="483" alt="image" src="https://github.com/user-attachments/assets/8cbca48c-af94-41bb-9973-b9731c23df14" />
<img width="727" height="758" alt="image" src="https://github.com/user-attachments/assets/10e3705a-7c8b-4384-ac18-d0ab20f1d241" />

### Phân tích dữ liệu

- Giá trị lớn nhất, nhỏ nhất:
<img width="1465" height="710" alt="image" src="https://github.com/user-attachments/assets/8fb3c135-e03f-497d-a8ed-1ba3a427815d" />
<img width="773" height="407" alt="image" src="https://github.com/user-attachments/assets/f7830be1-007c-49ce-acb9-6af4f39e0f1b" />

- Từ kết quả trên, ta có thể rút ra được kết luận:
+ age (tuổi)
Giá trị lớn nhất (max_age): 95
Giá trị nhỏ nhất (min_age): 18
Kết luận: Độ tuổi của các khách hàng trong tập dữ liệu dao động từ 18 đến 95 tuổi, cho thấy tập dữ liệu bao gồm cả người trẻ tuổi và người cao tuổi.

+ balance (số dư tài khoản)
Giá trị lớn nhất (max_balance): 102127
Giá trị nhỏ nhất (min_balance): -8019
Kết luận: Số dư tài khoản dao động từ -8019 đến 102127. Điều này cho thấy có khách hàng bị nợ (số dư âm) và có những khách hàng có số dư rất cao.

+ duration (thời lượng cuộc gọi)
Giá trị lớn nhất (max_duration): 4918
Giá trị nhỏ nhất (min_duration): 0
Kết luận: Thời lượng cuộc gọi dao động từ 0 đến 4918 giây (tương đương hơn 81 phút). Thời lượng bằng 0 có thể cho thấy các cuộc gọi không thành công hoặc bị từ chối ngay lập tức.

+ campaign (số lần liên hệ trong chiến dịch)
Giá trị lớn nhất (max_campaign): 63
Giá trị nhỏ nhất (min_campaign): 1
Kết luận: Số lần liên hệ trong chiến dịch dao động từ 1 đến 63 lần. Điều này cho thấy có khách hàng đã được liên hệ rất nhiều lần trong một chiến dịch.

+ pdays (số ngày kể từ khi khách hàng được liên hệ lần cuối trong chiến dịch trước)
Giá trị lớn nhất (max_pdays): 536
Giá trị nhỏ nhất (min_pdays): -1
Kết luận: Giá trị -1 có thể chỉ ra rằng khách hàng chưa từng được liên hệ trước đó. Số ngày kể từ lần liên hệ trước dao động từ 0 đến 536 ngày đối với những khách hàng đã được liên hệ trước đó.

+ previous (số lần liên hệ với khách hàng trước chiến dịch hiện tại)
Giá trị lớn nhất (max_previous): 275
Giá trị nhỏ nhất (min_previous): 0


### Support Vector Machine

![image](https://github.com/user-attachments/assets/e4aedb4c-7cd6-4835-8eba-b4da58909796)
![image](https://github.com/user-attachments/assets/e1b3fdf1-04de-4279-877d-28a398e79fd6)

### Song song hóa giải thuật dựa trên MapReduce
#### Ý tưởng

![image](https://github.com/user-attachments/assets/e5fc65bb-bb42-4e0a-ab27-b7d9ac524cfc)

#### Cài đặt

![image](https://github.com/user-attachments/assets/ac9889c3-44e8-49f3-a5da-04dfe5c12931)
![image](https://github.com/user-attachments/assets/e8a5c5f8-b03f-4906-8d5b-d2ad824958a6)
![image](https://github.com/user-attachments/assets/c9d47bb7-6d07-48e0-83d2-9dba5c844385)
![image](https://github.com/user-attachments/assets/810939fd-c39d-4342-b75c-25632ec21146)
![image](https://github.com/user-attachments/assets/de4f669c-7a64-4119-bde9-2c93395ed91c)
