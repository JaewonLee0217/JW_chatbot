# JW_chatbot
Python, Django, tensorflow


### Model
    * Model: "sequential_2"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      embedding_2 (Embedding)      (None, 20, 16)            16000     
      _________________________________________________________________
      global_average_pooling1d_2 ( (None, 16)                0         
      _________________________________________________________________
      dense_6 (Dense)              (None, 16)                272       
      _________________________________________________________________
      dense_7 (Dense)              (None, 16)                272       
      _________________________________________________________________
      dense_8 (Dense)              (None, 8)                 136       
      =================================================================
      Total params: 16,680
      Trainable params: 16,680
      Non-trainable params: 0
      _________________________________________________________________
      
      
     * Epoch -> 490
     * 최종 loss: 0.6715 - accuracy: 0.9388
     
### 개발 노트
![almost](https://user-images.githubusercontent.com/55820321/107842409-4ae03980-6e06-11eb-9fc7-bc583a1df934.JPG)

      * 채팅 환경 구축 완료 , 모델 로드 및 JWbot answering fuction 완료

      * Django 환경 구축 -> Flask사용 배포 (진행중)
      
      * 모델 성능 개선 연구중 (model on 21.02.13 = loss: 0.5963 - accuracy: 0.8775)
  
