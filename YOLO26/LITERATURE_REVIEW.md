# Literature review — vòng sàng lọc đầu tiên

Ngày: 2026-09-11. Trạng thái: **PARTIAL**, chưa đủ để xác nhận novelty. Đã kiểm tra nguồn sơ cấp ở CVF/arXiv/repository tác giả; chưa trích toàn bộ bảng/supplement, chưa rà đầy đủ 2026 và các journals yêu cầu. Đây là scoping review có cấu trúc, không được mô tả là systematic review hoàn tất.

Truy vấn theo các nhóm efficient convolution/PConv/reparameterization; FPN/PAN/BiFPN/adaptive fusion; real-time DETR 2025–2026; state-space/wavelet; KD/pruning/QAT/hardware NAS. Mở nguồn gốc theo title/ID, phân biệt năm preprint và publication, loại nguồn SEO/benchmark thiếu setting. Ba ID tra cứu không khớp title mục tiêu (2401.10125, 1903.05027, 2203.14983) đã bị loại; Vision Mamba, FCOS và FGD được tìm lại bằng title. Không dùng các ID sai làm evidence.

**NE = chưa trích/xác minh**, không có nghĩa paper không báo cáo. P/F/T lần lượt parameters/FLOPs/latency hoặc FPS. Nếu không ghi riêng, cả ba là NE, và accuracy chưa trích bảng. Nhận định tích hợp, hạn chế và risk là phân tích kỹ thuật, không phải kết quả trên YOLO11. Tác động của mọi đề xuất: **CHƯA ĐƯỢC XÁC MINH BẰNG THỰC NGHIỆM.**

## Backbone và convolution

| Nguồn / năm | Ý tưởng và lợi ích | Hạn chế; tích hợp YOLO11 | Nguy cơ trùng novelty nếu dùng trực tiếp | P/F/T và accuracy đã xác minh |
|---|---|---|---|---|
| [MobileNets](https://arxiv.org/abs/1704.04861), 2017 preprint | Depthwise spatial + pointwise mixing, giảm MAC | Memory access và kernel efficiency; head cls local đã dùng DW | HIGH | NE |
| [MobileNetV2](https://arxiv.org/abs/1801.04381), CVPR 2018 | Inverted residual, linear bottleneck | Expansion tăng activation; adapter P3–P5 khả thi | HIGH | NE |
| [MobileNetV4](https://arxiv.org/abs/2404.10518), ECCV 2024 | UIB, Mobile MQA, hardware NAS, distillation | Runtime EdgeTPU không chuyển sang RTX; local đã có họ MNV4 | HIGH | P/F NE; Hybrid-Large 3.8 ms Pixel8 EdgeTPU, 87% ImageNet top-1 theo abstract |
| [EfficientNet](https://arxiv.org/abs/1905.11946), ICML 2019 | Compound depth/width/resolution scaling | Không bảo đảm scaling đồng đều tối ưu neck/head | HIGH | NE |
| [GhostNet](https://arxiv.org/abs/1911.11907), preprint 2019 / CVPR 2020 | Sinh feature từ cheap operations | Concat/branches vẫn có cost; GhostConv đã có local | HIGH | NE |
| [ShuffleNet V2](https://arxiv.org/abs/1807.11164), ECCV 2018 | Split/shuffle, tối ưu memory access và fragmentation | Phải đo shuffle/concat, không suy tốc độ từ FLOPs | HIGH | NE |
| [FasterNet](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html), CVPR 2023 | PConv spatial-convolve một phần channels | Ratio nhỏ giới hạn spatial mixing; local có PConv/C3k2_Faster | HIGH | NE số tuyệt đối; paper so throughput trên nhiều thiết bị |
| [RepVGG](https://arxiv.org/abs/2101.03697), CVPR 2021 | Multi-branch train, single-conv deploy | Cần chứng minh BN/bias/padding equivalence; không tự giảm deploy kernel | HIGH | NE |
| [Diverse Branch Block](https://arxiv.org/abs/2103.13425), CVPR 2021 | Diverse linear branches quy về một conv | Nonlinearity giữa branch stages có thể ngăn exact fusion | HIGH | NE |
| [MobileOne](https://arxiv.org/abs/2206.04040), preprint 2022 / CVPR 2023 | Reparameterization và tối ưu latency thiết bị | iPhone benchmark không tương đương Windows CPU/GPU | HIGH | P/F NE; dưới 1 ms iPhone12, 75.9% ImageNet top-1 cho biến thể trong abstract |
| [ConvNeXt](https://arxiv.org/abs/2201.03545), CVPR 2022 | CNN hiện đại, kernel rộng, recipe cải tiến | Layout/normalization/expansion phải profile; pretrained confounder | HIGH | NE |
| [CSPNet](https://arxiv.org/abs/1911.11929), preprint 2019 / CVPRW 2020 | Phân luồng computation và gradient, rồi aggregate | YOLO11 vốn thuộc CSP family | HIGH | NE |
| [YOLOv9](https://arxiv.org/abs/2402.13616), 2024 | PGI và GELAN aggregation | Phải tách gain từ supervision và topology | HIGH | NE |
| [PP-LCNet](https://arxiv.org/abs/2109.15099), 2021 preprint | Thiết kế theo CPU/MKLDNN | Paddle latency không chuyển trực tiếp sang PyTorch | HIGH | NE |
| [PP-HGNetV2 source](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py), release/2.6 | Feature aggregation, projection, light blocks | Model release; chưa xác minh năm paper, không gán DOI; local HGBlock không phải cả backbone | HIGH | NE |
| [Dynamic Convolution](https://arxiv.org/abs/1912.03458), preprint 2019 / CVPR 2020 | Input-dependent kernel aggregation | Không exact-fuse thành kernel tĩnh cho mọi input; export phức tạp | HIGH | P/T NE; +4% FLOPs, +2.9 điểm top-1 với MobileNetV3-Small theo abstract |
| [RepLKNet](https://arxiv.org/abs/2203.06717), CVPR 2022 | Large depthwise kernels, reparameterization | Receptive field lớn không bảo đảm nhanh ở mọi resolution | HIGH | NE |
| [Deformable ConvNets](https://arxiv.org/abs/1703.06211), ICCV 2017 | Learned offsets thích ứng hình học | Irregular sampling/custom kernels; dùng khi errors hỗ trợ giả thuyết | HIGH | NE |

## Fusion, attention và representation

| Nguồn / năm | Ý tưởng và lợi ích | Hạn chế; tích hợp YOLO11 | Risk | P/F/T và accuracy |
|---|---|---|---|---|
| [FPN](https://arxiv.org/abs/1612.03144), preprint 2016 / CVPR 2017 | Top-down lateral fusion đưa semantics đến resolution cao | YOLO11 đã có đường top-down | HIGH | NE setting đầy đủ |
| [PANet](https://arxiv.org/abs/1803.01534), CVPR 2018 | Bottom-up aggregation và adaptive pooling | YOLO neck chỉ chung nguyên lý, không phải toàn bộ PANet | HIGH | NE |
| [EfficientDet/BiFPN](https://arxiv.org/abs/1911.09070), preprint 2019 / CVPR 2020 | Weighted bidirectional fusion, compound scaling, shared head towers | Static scale weights đã có prior art; resize/repeat overhead | HIGH | D7: 77M / 410B theo nguồn / T NE; 55.1 AP COCO test-dev, single-model single-scale |
| [ASFF](https://arxiv.org/abs/1911.09516), 2019 preprint | Spatially adaptive scale fusion | Input-dependent weights phải tính lúc inference | HIGH | NE setting đầy đủ; không dùng AP/FPS abstract thiếu hardware |
| [SENet](https://arxiv.org/abs/1709.01507), preprint 2017 / CVPR 2018 | Channel recalibration qua global pooling | Pooling mất locality; đã có SE-like blocks local | HIGH | NE |
| [CBAM](https://arxiv.org/abs/1807.06521), ECCV 2018 | Channel rồi spatial gating | Thêm activation passes; không tự giải quyết budget bottleneck | HIGH | NE |
| [ECA-Net](https://arxiv.org/abs/1910.03151), preprint 2019 / CVPR 2020 | Local channel interaction không MLP reduction | Ít params nhưng pooling/multiply vẫn có cost | HIGH | NE |
| [Coordinate Attention](https://arxiv.org/abs/2103.02907), CVPR 2021 | Pooling theo trục giữ positional information | Không mặc định cải thiện small objects; đã có CoordAtt local | HIGH | NE |
| [Linear attention](https://arxiv.org/abs/2006.16236), ICML 2020 | Associativity tránh N² token matrix | Constants/channel dimensions vẫn quyết định runtime; không phải detector cố định | HIGH | P/F/T detector không áp dụng; local accuracy NE |
| [FcaNet](https://arxiv.org/abs/2012.11879), preprint 2020 / ICCV 2021 | Multi-spectral channel attention | Frequency gate tự thân đã có prior art | HIGH | NE |
| [WTConv](https://arxiv.org/html/2407.05848v1), 2024 | Multilevel wavelet-band conv tăng receptive field | Transform/inverse có cost; không mặc nhiên fuse thành small conv | HIGH | NE; đã đọc method/limitations, chưa trích đủ bảng |
| [Low-rank expansions](https://arxiv.org/abs/1405.3866), 2014 | Factorize convolutions giảm computation | Rank thấp có thể mất information; nhiều launches | HIGH | NE |

## Detector, head và training/deployment

| Nguồn / năm | Ý tưởng và lợi ích | Hạn chế; tích hợp YOLO11 | Risk | P/F/T và accuracy |
|---|---|---|---|---|
| [DETR](https://arxiv.org/abs/2005.12872), ECCV 2020 | Queries, set prediction, bipartite matching, bỏ NMS | Thay cả head/loss; external comparator phù hợp hơn drop-in | HIGH | NE |
| [Deformable DETR](https://arxiv.org/abs/2010.04159), preprint 2020 / ICLR 2021 | Sparse multiscale attention | Sampling kernels/export cần kiểm tra | HIGH | NE |
| [RT-DETR](https://arxiv.org/abs/2304.08069), preprint 2023 / CVPR 2024 | Tách intra-scale interaction/cross-scale fusion; query selection | Có RTDETRDecoder local; cần kiểm soát data/pretraining | HIGH | P/F NE; R50/R101: 108/74 FPS T4, 53.1/54.3 COCO AP theo abstract; không so trực tiếp RTX4060 |
| [RT-DETRv2](https://arxiv.org/abs/2407.17140), 2024 preprint | Bag-of-freebies cải thiện baseline | Không gán gain training cho architecture | HIGH | NE |
| [MobileViT](https://arxiv.org/abs/2110.02178), preprint 2021 / ICLR 2022 | CNN local + transformer global representation | Unfold/fold/layout overhead; cần đo trước khi thay C2PSA | HIGH | khoảng 6M / F/T NE; 78.4% ImageNet top-1 theo abstract |
| [VMamba](https://arxiv.org/abs/2401.10166), 2024 | 2D selective scan state-space | Kernel support/export; linear asymptotic không bảo đảm nhanh ở 20×20 | HIGH | NE |
| [Vision Mamba](https://arxiv.org/abs/2401.09417), 2024 | Bidirectional state-space visual representation | Khác VMamba; không gọi linear conv bất kỳ là selective SSM | HIGH | NE |
| [FCOS](https://arxiv.org/abs/1904.01355), ICCV 2019 | Anchor-free dense regression, shared towers, centerness | Đối chứng cho shared head; YOLO11 đã anchor-free | HIGH | NE |
| [TOOD](https://arxiv.org/abs/2108.07755), ICCV 2021 | Task alignment giữa classification và localization | TAL đã có local; thêm task decomposition phải ablate riêng | HIGH | NE |
| [YOLOX](https://arxiv.org/abs/2107.08430), 2021 preprint | Decoupled head, anchor-free, SimOTA | Thay assignment tạo training confounder | HIGH | NE |
| [GFL](https://arxiv.org/abs/2006.04388), NeurIPS 2020 | Distribution regression và quality representation | DFL đã có trong YOLO11 | HIGH | NE |
| [DIoU/CIoU](https://arxiv.org/abs/1911.08287), preprint 2019 / AAAI 2020 | Center/shape geometry trong loss | Local BboxLoss đã dùng CIoU; đổi loss không tự giảm inference cost | HIGH | P/F/T inference không đổi nếu chỉ đổi loss; accuracy local NE |
| [FGD](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf), CVPR 2022 | Focal/global feature distillation | Teacher chỉ lúc train; cần baseline+KD đối chứng | HIGH | P/F/T inference theo student; số tuyệt đối NE |
| [Filter pruning](https://arxiv.org/abs/1608.08710), preprint 2016 / ICLR 2017 | Structured filter removal | Zeros không tự giảm latency; concat dependencies cần xử lý | HIGH | NE |
| [Integer-only quantization training](https://arxiv.org/abs/1712.05877), preprint 2017 / CVPR 2018 | Training cho integer arithmetic deployment | Calibration/backend support và quantization error phải đo | HIGH | NE |
| [ProxylessNAS](https://arxiv.org/abs/1812.00332), preprint 2018 / ICLR 2019 | Search theo task/hardware thực | Search budget lớn; có nguy cơ overfit hardware/val | HIGH | NE |
| [Once-for-All](https://arxiv.org/abs/1908.09791), preprint 2019 / ICLR 2020 | Train supernet, specialize subnet | Subnet ranking và recipe fairness cần xác minh | HIGH | NE |
| [SAHI](https://arxiv.org/abs/2202.06934), 2022 | Slicing inference/fine-tuning cho vật nhỏ | Nhiều tiles tăng workload; phải báo full-image latency | HIGH | P theo detector, F/T theo số tiles; số tuyệt đối NE |
| [YOLOv10](https://arxiv.org/abs/2405.14458), 2024 | Consistent dual assignments, NMS-free và efficiency design | Không bật end2end YOLO11 rồi coi tương đương | HIGH | NE |
| [D-FINE](https://arxiv.org/html/2410.13842v1), preprint 2024 / ICLR 2025 | Fine-grained distribution refinement, self-distillation | Prior art gần cho distribution/KD head; refinement vẫn có cost | HIGH | NE bảng đầy đủ |
| [DEIM](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf), preprint 2024 / CVPR 2025 | Dense one-to-one matching và matching-aware training | Aux supervision đã có nhiều prior art; không gán gain cho topology | HIGH | P/F NE; L/X 124/78 FPS T4, 54.7/56.5 AP theo paper; không phải local results |
| [RT-DETRv3](https://arxiv.org/abs/2409.08475), 2024 preprint | Hierarchical dense positive supervision | Đối chứng cho training-only complexity | HIGH | P/F/T NE; abstract R18 48.1 AP |
| [DEIMv2](https://arxiv.org/abs/2509.20787), 2025 preprint | DINOv3 pretrained/distilled backbones, spatial tuning adapter | Large pretraining confounder; không so với scratch YOLO11 như cùng setting | HIGH | X 50.3M / F/T NE; abstract 57.8 AP, setup extraction còn thiếu |
| [YOLO11 v8.3.0](https://github.com/ultralytics/ultralytics/tree/v8.3.0), 2024 model release | C3k2, C2PSA, lightweight classification branch | Local khác release; cần đối chiếu SPPF/head/init/loss | HIGH nếu chỉ tái tổ chức modules đã có | Local n đo 2,624,080 params; không gán local measurements cho upstream |

## Coverage và bước tiếp theo

Đã có nguồn cho 40 nhóm: lightweight/efficient/DW/Ghost/partial conv; rep/reparameterization/CSP/ELAN-GELAN; multiscale/BiFPN/PAN-FPN/adaptive fusion; dynamic/large kernel/deformable; efficient/channel/spatial/coordinate/linear attention; transformer/hybrid/SSM; frequency/wavelet/low-rank/structural reparameterization; lightweight/shared/decoupled/anchor-free head, assignment/IoU/distribution; KD/pruning/QAT/NAS; small objects và edge. **Coverage nguồn không đồng nghĩa đã hoàn tất review định lượng.**

YOLOv8 có YAML và code local nhưng chưa chạy; không gán paper riêng chưa xác minh. ImageNet top-1 không thay thế detector mAP. Các số ở các hàng trên chỉ là kết quả nguồn, không tạo leaderboard chung vì settings khác nhau.

Trước thiết kế cần: trích đủ variant/input/precision/batch/runtime/hardware/pretraining/AP split từ full-text của nearest works; kiểm tra citations tiến/lùi và 2025–2026/journals; đối chiếu các modules tương tự đã tồn tại trong `plans/` và `nn/modules/`. Không xem frequency decomposition, static weighted sum, shared head hay cộng linear branches tự thân là đóng góp mới. Chưa có proposed model nên chưa thể hoàn thành novelty matrix hoặc final decision một cách có ý nghĩa.
