# Machine-Learning
# FINAL PROJECT MACHINE LEARNING

Người thực hiện: **Trần Quang Luân - 52100254**

**BÀI 1**

**Đề bài**: Trình bày một bài nghiên cứu, đánh giá của em về các vấn đề sau:
1) Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy;
2) Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.

# Trình bày

**Câu 1:**
- Đặt vấn đề: Trong nghiên cứu này, tôi đã tìm hiểu và so sánh các phương pháp Optimizer phổ biến trong huấn luyện mô hình học máy. Các phương pháp Optimizer có vai trò quan trọng trong việc điều chỉnh các tham số của mô hình để tối ưu hóa hàm mục tiêu.
- Khái niệm: Optimizer (thuật toán tối ưu hóa) là cơ sở để xây dựng mô hình neural network với mục đích "học " được các features ( hay pattern) của dữ liệu đầu vào, từ đó có thể tìm 1 cặp weights và bias phù hợp để tối ưu hóa model.
- Các phương pháp Optimizer trong huấn luyện mô hình học máy: 
![image](https://github.com/botkttv2003/Machine-Learning/assets/105039417/90725b9a-fb69-4bbf-8cc0-b6bc3024acd3)

  **1. Gradient Descent (GD)**
    - Gradient Descent (GD) là phương pháp cơ bản nhất trong tối ưu hóa mô hình.
    - GD tính toán gradient của hàm mất mát (loss function) theo từng tham số của mô hình.
    - Sau đó, nó di chuyển các tham số ngược chiều gradient với một tỷ lệ học tập cố định.
      ![image](https://github.com/botkttv2003/Machine-Learning/assets/105039417/b0423b96-d63e-450b-b92d-12d6c358a270)

    - Công thức cập nhật cho GD: θ = θ - α * ∇J(θ), trong đó: 
      +  θ là vector các tham số, 
      + α là tỷ lệ học tập, 
      +  ∇J(θ) là gradient của hàm mất mát J theo θ.
  
  **2. Stochastic Gradient Descent (SGD)**
    - Stochastic Gradient Descent (SGD) là phiên bản ngẫu nhiên của GD.
    - Thay vì tính toán gradient trên toàn bộ tập dữ liệu (như GD), SGD chỉ tính toán gradient trên một mẫu dữ liệu ngẫu nhiên trong mỗi bước cập nhật.
    -  Điều này giúp giảm khối lượng tính toán và tốc độ huấn luyện.
      ![image](https://github.com/botkttv2003/Machine-Learning/assets/105039417/c48c055f-7ff9-4038-8172-58de0e8dda53)

    - Công thức cập nhật cho SGD: θ = θ - α * ∇J(θ, x(i)), trong đó:
      + x(i) là một mẫu ngẫu nhiên, 
      + i là chỉ số mẫu.
        
  **3. Momentum**
    - Momentum giúp tăng tốc quá trình hội tụ bằng cách tích lũy gradient của các bước trước đó và thêm một lượng momentum (động lượng) vào quá trình cập nhật.
    - Nó giúp vượt qua các địa phương cực tiểu và nhanh chóng tiến gần hơn đến điểm tối ưu.
      ![image](https://github.com/botkttv2003/Machine-Learning/assets/105039417/f40c42ef-6db8-4078-afc1-cd55adce00b3)

    - Công thức cập nhật cho Momentum: v(t) = β * v(t-1) + α * ∇J(θ), θ = θ - v(t), trong đó: 
      + v(t) là vector momentum tại thời điểm t, 
      + β là hệ số momentum (0 < β < 1).
        
  **4. Adagrad (Adaptive Gradient Algorithm)**
    - AdaGrad là phương pháp điều chỉnh tỷ lệ học tập tự động cho từng tham số dựa trên lịch sử của các gradient trước đó.
    - AdaGrad phù hợp cho các bài toán với dữ liệu thưa và có thay đổi tỷ lệ học tập cho từng tham số riêng biệt.
    - Công thức cập nhật cho AdaGrad: G(t) = G(t-1) + (∇J(θ))^2, θ = θ - α / √(G(t) + ε) * ∇J(θ), trong đó: 
      + G(t) là ma trận đường chéo chứa tổng bình phương gradient trước đó, 
      + ε là một hằng số nhỏ để tránh chia cho 0.
        
  **5. RMSprop (Root Mean Square Propagation)**
    - RMSprop kết hợp ưu điểm của AdaGrad và Momentum.
    - Nó điều chỉnh tỷ lệ học tập dựa trên trung bình gia quyền của các gradient gần đây.
    - Công thức cập nhật cho RMSprop: G(t) = α * G(t-1) + (1 - α) * (∇J(θ))^2, θ = θ - ε / √(G(t) + ε) * ∇J(θ), trong đó: 
      + G(t) là trung bình gia quyền của các bình phương gradient trước đó, 
      + α là hệ số trọng số (0 < α < 1), 
      + ε là một hằng số nhỏ để tránh
        
  **6. Adam (Adaptive Moment Estimation)**
    - Adam là một phương pháp tối ưu hóa tỷ lệ học tập (learning rate) thích ứng cho mô hình học máy.
    - Nó kết hợp cả hai khía cạnh của phương pháp Momentum và RMSprop.
    - Adam tính toán hai bước để cập nhật tham số: một bước sử dụng moment (động lượng) và một bước sử dụng trung bình gia quyền của gradient bình phương.
    - Công thức cập nhật cho Adam:
      + Bước 1: Tính toán moment không chuẩn hóa (uncentered moment): m(t) = β1 * m(t-1) + (1 - β1) * ∇J(θ), trong đó:
        
        + m(t) là moment tại thời điểm t,
        + β1 là hệ số momentum (0 < β1 < 1),
        + ∇J(θ) là gradient của hàm mất mát J theo tham số θ.
      + Bước 2: Tính toán moment chuẩn hóa (centered moment): v(t) = β2 * v(t-1) + (1 - β2) * (∇J(θ))^2, trong đó: 
        + v(t) là moment chuẩn hóa tại thời điểm t, 
        + β2 là hệ số trọng số gradient bình phương (0 < β2 < 1).
      + Bước 3: Cập nhật tham số: θ = θ - α * m(t) / (√(v(t)) + ε), trong đó:
        + θ là vector các tham số, 
        + α là tỷ lệ học tập, 
        + ε là một hằng số nhỏ để tránh chia cho 0.

**->  Tổng quan:**
  + Còn có rất nhiều thuật toán tối ưu như Nesterov (NAG), Adadelta, Nadam,... nhưng mình sẽ không trình bày trong bài này, mình chỉ tập trung vào các optimizers hay được sử dụng. Hiện nay optimizers hay được sử dụng nhất là 'Adam'.
    
    ![image](https://github.com/botkttv2003/Machine-Learning/assets/105039417/a17c3e7b-fe3d-4e5a-8ca3-3b9295a93fb9)
    
  + Qua hình trên ta thấy optimizer 'Adam' hoạt động khá tốt, tiến nhanh tới mức tối thiểu hơn các phương pháp khác. 

**So sánh các phương pháp Optimizer:**

| Optimizer | Ưu điểm | Nhược điểm |
| :---: | :---: | :---: |
| Gradient Descent | - Dễ hiểu và triển khai.<br>- Tính toán đơn giản và phổ biến.<br>- Hoạt động tốt trên tập dữ liệu lớn khi được kết hợp với các kỹ thuật như mini-batch. | Có thể chậm trong việc hội tụ đến điểm tối ưu, đặc biệt trên bề mặt hàm không gồ ghề hoặc khi tỷ lệ học (learning rate) không được chọn tốt. |
| Stochastic Gradient Descent | - Phù hợp với dữ liệu lớn vì chỉ cần tính gradient trên một mẫu dữ liệu (mini-batch) thay vì toàn bộ dữ liệu.<br>- Có thể hội tụ nhanh hơn so với Gradient Descent truyền thống. | - Gây độ dao động cao hơn Gradient Descent truyền thống vì gradient được ước lượng dựa trên một mẫu dữ liệu nhỏ.<br>- Cần điều chỉnh kỹ thuật tốn thời gian để điều chỉnh tỷ lệ học (learning rate). |
| Momentum | - Giúp tăng tốc quá trình hội tụ và giảm độ dao động của gradient.<br>- Có khả năng vượt qua điểm tối ưu cục bộ và đi đến điểm tối ưu toàn cục nhanh hơn. | - Cần điều chỉnh tham số động lượng (momentum) và tỷ lệ học (learning rate) để đạt hiệu quả tốt nhất.<br>- Đôi khi có thể quá tối ưu và dẫn đến vấn đề overshooting (vượt quá điểm tối ưu). |
| Adagrad | - Tự điều chỉnh tỷ lệ học (learning rate) cho từng tham số dựa trên bình phương của gradient.<br>- Hiệu quả cho các tham số có gradient lớn và thưa (sparse gradient). | - Tích lũy squared gradient có thể dẫn đến việc giảm tỷ lệ học quá nhanh, làm cho quá trình học chậm dần lại.<br>- Khó khăn trong việc điều chỉnh tỷ lệ học tự động. |
| RMSprop | - Hiệu quả trong việc xử lý các đặc trưng có thay đổi đáng kể trong dữ liệu.<br>- Tự điều chỉnh tỷ lệ học (learning rate) cho từng tham số dựa trên trung bình của squared gradient. | Cần điều chỉnh tham số tỷ lệ học (learning rate) để đạt hiệu quả tốt nhất. |
| Adam | - Kết hợp sự hiệu quả của Momentum và RMSprop.<br>- Có khả năng tìm được điểm tối ưu nhanh chóng và ổn định trên nhiều loại hàm mục tiêu. | - Cần điều chỉnh nhiều tham số và tỷ lệ học (learning rate).<br>- Tích lũy sử dụng bộ nhớ và tính toán phức tạp hơn các phương pháp khác. |

**Câu 2:**
- Continual Learning:
  + Để điều chỉnh mô hình của chúng ta đối với sự thay đổi trong 	phân phối dữ liệu
  + Các vấn đề về cơ sở hạ tầng
  + Mục đích: Để tự động hóa việc cập nhật mô hình một cách an 	toàn và hiệu quả	
- Test Production:
	+ Mô hình được huấn luyện lại để thích ứng với môi trường thay 		đổi
		- Đánh giá nó trên một tập dữ liệu cố định
		- Cũng kiểm thử trong môi trường sản xuất
	+ Theo dõi và kiểm thử trong môi trường sản xuất
		- Theo dõi: theo dõi các đầu ra một cách tích cực
		- Kiểm thử trong môi trường sản xuất: chọn mô hình nào để tạo ra đầu ra một cách tích cực
	+ Mục đích: Hiểu hiệu suất của một mô hình và xác định khi nào nên cập nhật nó

**1. Continual Learning**
- Đặt vấn đề: Trong quá trình triển khai giải pháp học máy, việc đối mặt với dữ liệu mới và mở rộng bài toán là điều thường xuyên xảy ra. Continual Learning đảm bảo rằng mô hình được cập nhật và thích ứng với dữ liệu mới, đồng thời giữ lại kiến thức đã học để không phải huấn luyện lại từ đầu.
- Khái niệm: Continual Learning (học liên tục) là khả năng của một hệ thống học máy để liên tục học và tích lũy kiến thức mới mà không quên đi kiến thức đã học trước đó. 
- Ưu điểm của Continual Learning:
	+ Khả năng tiếp nhận dữ liệu mới và mở rộng bài toán.
	+ Tiết kiệm thời gian và công sức so với việc huấn luyện lại từ đầu.
- Nhược điểm của Continual Learning:
	+ Có thể gặp vấn đề quên mất kiến thức đã học trước đó.
	+ Đòi hỏi quản lý bộ nhớ và kiến thức trước đó của mô hình.
- Các bước triển khai:
	+ Giai đoạn 1: Huấn luyện lại thủ công, không lưu trạng thái
	+ Giai đoạn 2: Huấn luyện lại tự động
	+ Giai đoạn 3: Huấn luyện lại tự động, lưu trạng thái
	+ Giai đoạn 4: Học liên tục
- Các phương pháp triển  khai: 
	+ Elastic Weight Consolidation (EWC): Phương pháp này đo lường sự quan trọng của các trọng số trong mô hình và giữ lại những trọng số quan trọng đối với các nhiệm vụ trước đó. Điều này giúp đảm bảo rằng kiến thức đã học không bị mất đi khi huấn luyện mô hình cho các nhiệm vụ mới.
	+ Online EWC: Tương tự như EWC, nhưng thay vì chỉ tính toán sự quan trọng của các trọng số khi huấn luyện một nhiệm vụ xong, phương pháp này tính toán sự quan trọng ngay trong quá trình huấn luyện, cho phép mô hình nắm bắt kiến thức mới từ dữ liệu đang xử lý.
	+ Generative Replay: Phương pháp này sử dụng một mạng sinh để tạo ra các mẫu dữ liệu giả mạo từ các nhiệm vụ trước đó. Mô hình huấn luyện trên dữ liệu mới kết hợp với dữ liệu giả mạo này để giữ lại kiến thức đã học.

**2. Test Production**
- Đặt vấn đề: Khi xây dựng giải pháp học máy, việc đánh giá mô hình là một bước quan trọng để đảm bảo rằng nó hoạt động chính xác và đáng tin cậy trên dữ liệu thực tế.
- Khái niệm: Test Production (sản xuất kiểm tra) là quy trình đánh giá hiệu suất của mô hình học máy trên dữ liệu mới hoặc dữ liệu thực tế. 
- Ưu điểm của Test Production:
	+ Xác định hiệu suất thực tế của mô hình trên dữ liệu mới.
	+  Đánh giá tính đáng tin cậy và khả năng tổng quát hóa của mô hình.
- Nhược điểm của Test Production:
	+  Đòi hỏi dữ liệu kiểm tra chất lượng và đại diện cho thực tế.
	+  Đòi hỏi thời gian và tài nguyên để thực hiện quá trình kiểm tra.
- Các  phương pháp triển khai:
	+ Hold-Out Validation: Phương pháp này chia dữ liệu thành hai phần: tập huấn luyện và tập kiểm tra (hold-out set). Mô hình được huấn luyện trên tập huấn luyện và sau đó được đánh giá trên tập kiểm tra để đo lường hiệu suất.
	+ Cross-Validation: Phương pháp này chia dữ liệu thành k-fold subsets. Mô hình được huấn luyện k lần, mỗi lần sử dụng k-1 subsets để huấn luyện và subset còn lại để kiểm tra. Điều này giúp đánh giá mô hình trên nhiều tập kiểm tra khác nhau và giảm khả năng overfitting.
	+ A/B Testing: Phương pháp này được sử dụng trong các tình huống triển khai hệ thống thực tế. Hai phiên bản (A và B) của giải pháp học máy được triển khai song song và đánh giá hiệu suất bằng cách so sánh kết quả của họ trên dữ liệu thực tế.

**-> Tổng kết:** Continual Learning giúp mô hình học máy tiếp nhận dữ liệu mới và mở rộng bài toán một cách liên tục, đồng thời giữ lại kiến thức đã học. Trong khi đó, Test Production đảm bảo đánh giá hiệu suất của mô hình trên dữ liệu mới hoặc dữ liệu thực tế. Cả hai khía cạnh này đều quan trọng và cần được xem xét trong quá trình xây dựng một giải pháp học máy thành công.

#  TÀI LIỆU THAM KHẢO

**Tiếng Việt**

1. Machine Learning cơ bản (machinelearningcoban.com)<br>
2. Optimizer- Hiểu sâu về các thuật toán tối ưu ( GD,SGD,Adam,..) (viblo.asia)

**Tiếng Anh**

3. Optimizers — ML Glossary documentation (ml-cheatsheet.readthedocs.io)<br>
4. Chapter 9. Continual Learning and Test in Production (aiden-jeon.github.io)<br>
5. Continual Learning | Papers With Code<br>
6. Testing in Production: What It Is & Why You Should Do It | Perfecto by Perforce
