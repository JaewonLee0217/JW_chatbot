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

      *기본적인 JWchat방 구축 완료 -> 세부 javascript 작성중
      ![초기화면](https://user-images.githubusercontent.com/55820321/107602078-d6719300-6c6b-11eb-8777-47bce768ef3e.JPG)


      * Django 환경 구축 -> Flask사용 배포 (진행중)
      
      * 모델 성능 개선 연구중 (진행중)
  
