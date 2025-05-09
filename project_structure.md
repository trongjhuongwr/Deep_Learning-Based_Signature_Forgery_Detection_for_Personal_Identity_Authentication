│
├── models/
│   ├── feature_extractor.py        # Chứa CNN ResNet34 
│   ├── triplet_network.py           # TripletNetwork class
│   ├── siamese_network.py           # SiameseNetwork class
│
├── losses/
│   ├── triplet_loss.py              # Hàm loss cho Triplet Network
│   ├── contrastive_loss.py          # Hàm loss cho Siamese Network
│
├── datasets/
│   ├── triplet_dataloader.py           # Dataset class cho Triplet loader, load data cho train
│   ├── siamese_dataloader.py           # Dataset class cho Siamese loader, load data cho train
│
├── utils/
│   ├── metrics.py                   # accuracy, distance, F1, ....
│   ├── helpers.py                   # Hàm hỗ trợ cho việc train dễ hơn (ví dụ: load checkpoint, visualize)
│
├── configs/
│   ├── config_triplet.yaml          # File cấu hình riêng cho Triplet
│   ├── config_siamese.yaml          # File cấu hình riêng cho Siamese
│
├── train_models/
│   ├── train_triplet.ipynb          # Train Triplet
│   ├── train_siamese.ipynb          # Train Siamese
│
├── test_models/
│   ├── triplet_test.ipynb          # Test Triplet
│   ├── siamese_test.ipynb          # Test Siamese
│
├── README.md                        
└── requirements.txt                 
