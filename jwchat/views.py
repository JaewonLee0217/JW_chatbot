from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Create your views here.
#홈에서 요청하면 jwchathome을 부른다
def home(request):
    context = {}
    return render(request,"jwchathome.html",context)

@csrf_exempt
def chattrain(request):
    context = {}

    file = open(f"./static/intents.json")
    data = json.load(file)
    # 데이터 구조는 패턴, 리스폰스, 태그로 구성되어 있고,

    training_sentences = []  # 실제 어구
    training_labels = []  # 태그 정보들
    labels = []  # 이 문장이 인사인지 도움인지 뭐를 뜻하는 지 태그 정보
    responses = []  # 패턴에 대한 반응

    # data proprecessing -> 각 리스트에 정보 분배
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        # 라벨 리스트에 태그 정보들 기록
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # 라벨 리스트에 담은 길이가 클래스의 총 개수로 정해준다.
    num_classes = len(labels)

    # 인코딩 실행 sklearn의 전처리 라이브러리 불러오서 각 패턴에 대한 태그들,
    # 뭐라뭐라 말했을 떄 그 말에 대한 단어들을 모아둔 training_labels 에서 인코더를 fit 시키고
    # 그것을 가주고 숫자 형태로 transforming 시켜 준다.
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    # tokenizer는 텐서플로우의 keras라이브러리 꺼 사용.
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    # 여기서 문장에서 각 토큰 들 짤라주고
    tokenizer.fit_on_texts(training_sentences)

    # 짜른 토큰들을 word to index화 시킨다.
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)

    # 그럼 sequence에는 각 단어 문장의 단어마다 숫자로 표현되어 있지만 길이가
    # 서로 다르므로 제로 패딩을 통해서 각 문장마다 길이 동일하게 만들어주고 트레이닝을 진행해야 한다
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    # 모델 훈련
    # 모델을 먼저 텐서플로우 케라스 모델에서 세퀀셜가져오고

    # keras.layer 라이브러리에서 임베딩함수로 인풋이 들어갈 부분을 만들어준다.
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    epochs = 490
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("static/JW_model_2")



    # to save the fitted tokenizer
    with open('static/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open('static/label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    #ajax에서 이거를 받아서 result로 쓰게 된다.
    context["result"] = "Success"

    #ajax로 받기로
    return JsonResponse(context, content_type="application/json")


@csrf_exempt
def chatanswer(request):
    context = {}

    questext = request.GET['questext']

    import colorama
    colorama.init()
    from colorama import Fore, Style, Back
    import random


    file = open(f"./static/intents.json")
    data = json.load(file)

    def chat3(inp):
        # load trained model
        model = keras.models.load_model('static/JW_model_2')

        # load tokenizer object
        with open('static/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('static/label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20

        # while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        # inp = 'What is name'

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                txt1 = np.random.choice(i['responses'])
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, txt1)

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

        return txt1

    anstext = chat3(questext)
    print(anstext)
    context['anstext'] = anstext




    context["result"] = "Success"
    context["flag"] = 0


    # ajax로 받기로
    return JsonResponse(context, content_type="application/json")

