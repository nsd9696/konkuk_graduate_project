# Dehazing-Transformer
- 건국대학교 컴퓨터공학부 졸업 작품
- 구성원: 남상대, 이태영, 장효진

## Data
- “AOD-Net: All-in-One Dehazing Network”논문에서 구축한 데이터셋을 사용
- 총 1449쌍의 원본 이미지와 hazy 이미지 데이터를 사용하였으며 6:3:1의 비율로 train/valid/test 데이터셋으로 구축
- (샘플 데이터셋 이미지)

## Model 
- IPT(Image Processing Transformer) 레퍼런스 모델 참고
- (IPT 이미지)
- Knowledge Distillation 적용 예정
- (KD 이미지)

## Train
- optimizer: Adam
- learning rate: 1e-6
- Drop out: 0.2
- Metric: Mean Squared Error

## Product
- Application Framework: Streamlit
- CI/CD: Travis
- Model and Data are stored in Google Drive
-  

 
