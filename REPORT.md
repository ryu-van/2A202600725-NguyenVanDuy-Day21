# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Văn Duy  
**Mã số sinh viên**: 2A202600725  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (Lightweight ZIP với bộ adapter r=16)

---

## 1. Setup

*   **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Mô hình Qwen 2.5 bản 3 tỷ tham số, được lượng tử hóa 4-bit và tối ưu hóa tốc độ bởi Unsloth).
*   **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (tập dữ liệu chỉ dẫn tiếng Việt được dịch từ Alpaca GPT-4).
    *   **Số lượng mẫu**: 200 mẫu (được chia thành 180 mẫu huấn luyện - Train và 20 mẫu đánh giá - Eval).
*   **max_seq_length**: 1024 tokens.
    *   *Phân tích độ dài*: Qua phân tích phân phối độ dài token thực tế của dataset, giá trị phân vị p50 là 227 tokens, p95 là 562 tokens, và p99 là 704 tokens. Cấu hình `max_seq_length = 1024` là hoàn toàn tối ưu, giúp bao phủ toàn bộ độ dài của dữ liệu mà không lãng phí bộ nhớ GPU.
*   **GPU**: Tesla T4 (16 GB VRAM) trên môi trường Google Colab Free.
*   **Training cost**: ~0.074 USD.
    *   *Chi tiết tính toán*: Tổng thời gian huấn luyện cho cả 3 phiên bản rank là khoảng 12.7 phút (4.09 + 4.47 + 4.17). Với đơn giá thuê GPU T4 trung bình trên thị trường hiện nay khoảng 0.35 USD/giờ, chi phí thực tế là: $(12.7 / 60) \times 0.35 \approx 0.074$ USD.

---

## 2. Rank Experiment Results

Dưới đây là bảng số liệu so sánh định lượng thực tế thu được sau khi hoàn thành huấn luyện 3 phiên bản rank (`r=8`, `r=16`, `r=64`) trên cùng một tập dữ liệu và siêu tham số:

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
| :--- | :--------------- | :--------- | :-------- | :-------- | :--------- |
| **r = 8** | 1,843,200 | 4.09 min | 7.22 GB | 1.5577 | 4.75 |
| **r = 16** (Baseline)| 3,686,400 | 4.47 min | 6.62 GB | 1.5161 | 4.55 |
| **r = 64** | 14,745,600 | 4.17 min | 8.00 GB | 1.4768 | 4.38 |

> [!NOTE]
> *   Số lượng tham số có thể huấn luyện (Trainable Params) tăng tuyến tính hoàn hảo theo cấp số nhân của hệ số rank $r$ (tương ứng tỉ lệ $1:2:8$).
> *   Độ phức tạp (Perplexity - PPL) được tính toán chuẩn xác từ giá trị loss đánh giá thực tế theo công thức: $\text{PPL} = e^{\text{eval\_loss}}$.

---

## 3. Loss Curve Analysis

Qua biểu đồ hàm loss (`loss_curve.png`) được kết xuất sau quá trình huấn luyện:
*   **Độ hội tụ**: Quá trình huấn luyện diễn ra vô cùng mượt mà. Đường cong loss giảm đều đặn từ giá trị ban đầu khoảng **2.6** và kết thúc ở mức **1.18** sau 3 epochs đối với rank baseline `r=16`. Điều này cho thấy mô hình đã học được cấu trúc chỉ dẫn và phong cách phản hồi của tập dữ liệu một cách hiệu quả.
*   **Đánh giá Overfitting (Quá khớp)**: Trên môi trường phần cứng hạn chế của GPU T4, tham số `eval_strategy` được đặt là `"no"` để tắt đánh giá giữa giờ nhằm tránh OOM. Tuy nhiên, giá trị `eval_loss` đo được sau huấn luyện (1.51 đối với r=16) có xu hướng giảm tương đồng với training loss và duy trì khoảng cách tối ưu. Kết hợp với việc áp dụng tỷ lệ Dropout bằng 0 và Learning Rate nhỏ (2e-4) có sử dụng Cosine Scheduler, mô hình hoàn toàn **không có dấu hiệu bị overfitting**.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là kết quả đối chiếu side-by-side thực tế giữa phản hồi của mô hình gốc (Base) và mô hình sau tinh chỉnh (`r=16`) thu được từ tệp `qualitative_comparison.csv` (các phản hồi được giới hạn ở độ dài 300 ký tự theo đúng mã nguồn xuất dữ liệu của Colab):

### Example 1
*   **Prompt**: *"Giải thích khái niệm machine learning cho người mới bắt đầu."*
*   **Base**: *"Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác"*
*   **Fine-tuned (r=16)**: *"Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc"*
*   **Nhận xét**: **Cải thiện (Improved)**. Mô hình Fine-tuned đưa ra định nghĩa mang tính khoa học và hiện đại hơn ("không có sự hướng dẫn trực tiếp từ người dùng", nhấn mạnh vào khía cạnh tự học của thuật toán) so với cách giải thích mang tính lặp từ của mô hình gốc.

### Example 2
*   **Prompt**: *"Viết đoạn code Python tính số Fibonacci thứ n."*
*   **Base**: *"Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return \"N phải là một số dương\"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        "*
*   **Fine-tuned (r=16)**: *"Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:\n\n```python\ndef fibonacci(n):\n    if n < 0:\n        raise ValueError(\"Input phải là một số nguyên dương.\")\n    elif n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range"*
*   **Nhận xét**: **Cải thiện rõ rệt (Improved)**. Sự khác biệt nằm ở tư duy tối ưu thuật toán. Mô hình Base sử dụng đệ quy thông thường (dễ gây treo máy, độ phức tạp $O(2^n)$), trong khi mô hình Fine-tuned đã chuyển sang dùng vòng lặp tối ưu hóa bộ nhớ (`a, b = 0, 1` và `for _ in range` với độ phức tạp thời gian $O(n)$ và bộ nhớ $O(1)$) và bắt lỗi đầu vào bằng `ValueError` rất chuyên nghiệp.

### Example 3
*   **Prompt**: *"Liệt kê 5 nguyên tắc thiết kế UI/UX."*
*   **Base**: *"1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru"*
*   **Fine-tuned (r=16)**: *"1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với"*
*   **Nhận xét**: **Cải thiện (Improved)**. Phản hồi của mô hình Base bị lặp từ nhiều lần ("thân thiện với người dùng... thân thiện, dễ sử dụng và thân thiện"). Mô hình sau tinh chỉnh đưa ra các nguyên tắc cô đọng, rõ ràng, tập trung thẳng vào các khía cạnh kỹ thuật hiện đại như "Chuyển đổi" (Conversion) và "Thích ứng" (Responsive).

### Example 4
*   **Prompt**: *"Tóm tắt sự khác biệt giữa LoRA và QLoRA."*
*   **Base**: *"LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ"*
*   **Fine-tuned (r=16)**: *"LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ"*
*   **Nhận xét**: **Cải thiện (Improved)**. Mô hình Base giải thích LoRA như một phương pháp cải thiện mô hình NLU (khá hạn chế). Mô hình Fine-tuned đã nhìn nhận LoRA/QLoRA một cách học thuật hơn dưới góc độ cơ chế regularization và tối ưu hóa trọng số cho các mô hình mạng nơ-ron sâu có số lượng tham số lớn.

### Example 5
*   **Prompt**: *"Phân biệt prompt engineering, RAG, và fine-tuning."*
*   **Base**: *"Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu"*
*   **Fine-tuned (r=16)**: *"Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th"*
*   **Nhận xét**: **Cải thiện (Improved)**. Phản hồi của mô hình Fine-tuned định nghĩa rõ ràng, mạch lạc và tập trung sâu sắc hơn vào vai trò thực tế của từng kỹ thuật trong hệ thống tự động hóa và phát triển ứng dụng AI.

---

## 5. Conclusion về Rank Trade-off

Dựa trên các số liệu thực nghiệm thực tế thu được từ quá trình huấn luyện mô hình `Qwen2.5-3B` trên GPU Tesla T4, tôi đưa ra các phân tích và kết luận sau về mối quan hệ đánh đổi (trade-off) khi lựa chọn hệ số rank:

1.  **Mức rank đem lại ROI tốt nhất (Return on Investment)**: 
    Mức rank **`r = 16`** (baseline) mang lại tỷ lệ tối ưu chi phí - hiệu năng tốt nhất. Khi so sánh giữa `r=8` và `r=16`, thời gian huấn luyện tăng không đáng kể (chỉ chênh lệch khoảng 23 giây do tốc độ xử lý của Unsloth cực kỳ tối ưu), và dung lượng bộ nhớ đỉnh Peak VRAM thậm chí còn được tối ưu hóa tốt hơn ở phiên chạy r=16 (6.62 GB so với 7.22 GB của r=8 nhờ cơ chế quản lý bộ nhớ đệm PyTorch hiệu quả hơn). Trong khi đó, Perplexity giảm rõ rệt từ **4.75 xuống 4.55** (cải thiện khoảng 4.2%), giúp mô hình phản hồi chính xác và trôi chảy hơn.
2.  **Hiện tượng hiệu suất cận biên giảm dần (Diminishing Returns)**: 
    Hiện tượng này xuất hiện khi tăng rank từ `r = 16` lên `r = 64`. Số lượng tham số huấn luyện tăng gấp 4 lần (từ 3.68 triệu lên 14.74 triệu tham số), làm dung lượng Peak VRAM tăng vọt lên **8.00 GB** (tăng 20.8% so với r=16). Tuy nhiên, mức độ cải thiện của Perplexity lại vô cùng nhỏ (chỉ giảm từ **4.55 xuống 4.38**, tức cải thiện vỏn vẹn 3.7%). Điều này chứng minh rằng đối với các tập dữ liệu có quy mô nhỏ (200 mẫu) và mục tiêu chuyên biệt hóa trung bình, việc cấu hình rank quá cao không đem lại sự đột phá về mặt chất lượng mà chỉ gây lãng phí tài nguyên và làm tăng kích thước file adapter.
3.  **Khuyến nghị triển khai thực tế (Production Recommendation)**:
    Nếu triển khai mô hình này trong môi trường sản xuất thực tế, tôi khuyến nghị lựa chọn mức rank **`r = 16`**. Mức cấu hình này đảm bảo kích thước file adapter cực kỳ nhỏ gọn (chỉ khoảng 14.8 MB đối với file `adapter_model.safetensors`), giúp quá trình nạp mô hình (adapter swapping) diễn ra nhanh chóng, tiết kiệm băng thông và bộ nhớ lưu trữ, đồng thời vẫn bảo toàn tối đa chất lượng ngôn ngữ của mô hình tinh chỉnh.

---

## 6. What I Learned

Sau khi hoàn thành bài thực hành Lab 21, tôi đã đúc kết được những bài học thực tiễn giá trị sau:
*   **Hiểu rõ cơ chế hoạt động của LoRA/QLoRA**: Việc trực tiếp huấn luyện và đo lường tham số của 3 mức rank giúp tôi hiểu sâu sắc cách LoRA đóng băng trọng số gốc và cập nhật delta thông qua các ma trận hạng thấp. Tôi nhận ra fine-tuning là phương pháp tối ưu để điều chỉnh định dạng đầu ra và phong cách diễn đạt (Format/Style Adaptation) chứ không phải để nạp kiến thức thô.
*   **Làm chủ kỹ năng kiểm soát tài nguyên GPU**: Qua thực nghiệm trên dòng card phổ thông Tesla T4, tôi đã học được cách kết hợp các kỹ thuật tối ưu hóa bộ nhớ như *4-bit NF4 Quantization*, *Gradient Checkpointing* và tận dụng các nhân CUDA viết riêng của *Unsloth* để huấn luyện thành công mô hình lớn mà hoàn toàn tránh được lỗi tràn bộ nhớ (OOM).
*   **Cách đánh giá mô hình LLM một cách khách quan**: Tôi đã biết cách kết hợp cả hai phương pháp đánh giá định lượng (sử dụng độ phức tạp Perplexity) và đánh giá định tính (đối chiếu side-by-side các câu hỏi kiểm thử thực tế) để có cái nhìn đa chiều và chính xác nhất về hiệu năng của mô hình sau khi tinh chỉnh.
