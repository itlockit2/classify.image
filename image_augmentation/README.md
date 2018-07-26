# Image Augmentation Process

## Docker에 Train 이미지데이터 전송 
* Train 이미지는 original_image에 라벨별로 정리를 해 놓는다.
* sudo docker cp 를 이용하여 호스트 이미지를 도커에 전송
	* sudo docker cp /home/banana/image_augmentation/original_image/Hyun_soo 9841e190124b:/root/openface/training-images

## Docker에서 Train 이미지를 aligned_images로 생성
* ./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96

## Docker에서 생성한 aligned_images를 호스트로 전송
* aligned_images는 augmentation에 라벨별로 정리를 해 놓는다.
	* sudo docker cp 9841e190124b:/root/openface/aligned-images ~/image_augmentation/
* original_image중 일부분은 aligned_images가 생성되지 않는다.

## aligned_images 이미지들을 데이터 증강 작업 실시
* augmentation 폴더 안에 있는 이미지 증강
* generator_aug.ipynb 파일을 통해 Image Augmentation 실행
* grid 폴더에는 각 사진의 64개의 필터를 적용한 사진들이 8*8 grid 형태로 결합되어 들어있다.
* grid_after 폴더에는 8*8 grid 형태로 결합된 사진들이 64개로 분리되어 저장이 된다.

## grid_after 폴더에 있는 증강된 이미지들을 docker로 전송한다.
* sudo docker cp /home/banana/image_augmentation/grid_after/Hyun_soo 9841e190124b:/root/openface/training-images

## Docker 에서 다시 학습을 시킨다.
* 먼저 기존에 있던 aligned-images에 있는 데이터들을 지우고  학습을 시작한다.
	* ./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
* 이전과 마찬가지로 많은 데이터들이  aligned_images가 생성되지 않는다.
*  aligned_images에 있는 각 사진들에서 128개의 특이점을 추출한다.
	* ./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
* 생성된 csv 파일을 호스트로 전송한다.
	* sudo docker cp 9841e190124b:/root/openface/generated-embeddings ~/image_augmentation/

## Host에서 학습을 시작한다.
* 먼저 csv file을 읽는다.
```
df = pd.read_csv("/home/banana/image_augmentation/reps.csv")
X = df.iloc[:,:].values
df_2 = pd.read_csv("/home/banana/image_augmentation/labels.csv")
Y = df_2.iloc[:,0].values
```

* Training Data 과 Testing Data를 분리하고 CNN 학습을 위해 16*8 형태로 reshape 해준다.
```
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)
X_train = X_train.reshape(-1,16,8,1)
X_test = X_test.reshape(-1,16,8,1)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
```

* 학습을 시작한다.
```
model = Sequential()
model.add(Conv2D(12, kernel_size=(2, 2), activation='relu', padding='same', 
                 init='glorot_uniform',input_shape=(16, 8,1)))
# out put  = 16 * 8 , 12
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# out put = 8 * 4 , 12
model.add(Conv2D(24, kernel_size=(2, 2), activation='relu', padding='same',
          init='glorot_uniform', input_shape = (8,4,12)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# out put = 4 * 2 , 24
model.add(Flatten())
model.add(Dense(192, init='glorot_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))
                
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
