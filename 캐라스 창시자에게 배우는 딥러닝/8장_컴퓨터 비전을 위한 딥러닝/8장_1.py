## 05.08
    # 합성곱 신경망으로 MNIST 데이터 셋 맞추기

    # step 1) 간단한 커브넷 만들기
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(28,28,1))
x = layers.Conv2D(filters=32,kernel_size=3,activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64,kernel_size=3,activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128,kernel_size=3,activation='relu')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

    # step 2) MNIST 데이터로 훈련하기
from tensorflow.keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype("float32") / 255
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size=64)

    # step 3) 모델 평가
test_loss,test_acc = model.evaluate(test_images,test_labels)
print(f"테스트 정확도 : {test_acc:.3f}")