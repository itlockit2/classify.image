#Image Augmentation Process

Docker�� Train �̹��������� ���� 
* Train �̹����� original_image�� �󺧺��� ������ �� ���´�.
* sudo docker cp �� �̿��Ͽ� ȣ��Ʈ �̹����� ��Ŀ�� ����
	* sudo docker cp /home/banana/image_augmentation/original_image/Hyun_soo 9841e190124b:/root/openface/training-images

Docker���� Train �̹����� aligned_images�� ����
* ./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96

Docker���� ������ aligned_images�� ȣ��Ʈ�� ����
* aligned_images�� augmentation�� �󺧺��� ������ �� ���´�.
	* sudo docker cp 9841e190124b:/root/openface/aligned-images ~/image_augmentation/

## original_image�� �Ϻκ��� aligned_images�� �������� �ʴ´�.

aligned_images �̹������� ������ ���� �۾� �ǽ�
* augmentation ���� �ȿ� �ִ� �̹��� ����
* generator_aug.ipynb ������ ���� Image Augmentation ����
* grid �������� �� ������ 64���� ���͸� ������ �������� 8*8 grid ���·� ���յǾ� ����ִ�.
* grid_after �������� 8*8 grid ���·� ���յ� �������� 64���� �и��Ǿ� ������ �ȴ�.

grid_after ������ �ִ� ������ �̹������� docker�� �����Ѵ�.
* sudo docker cp /home/banana/image_augmentation/grid_after/Hyun_soo 9841e190124b:/root/openface/training-images

Docker ���� �ٽ� �н��� ��Ų��.
* ���� ������ �ִ� aligned-images�� �ִ� �����͵��� �����  �н��� �����Ѵ�.
	* ./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
* ������ ���������� ���� �����͵���  aligned_images�� �������� �ʴ´�.
*  aligned_images�� �ִ� �� �����鿡�� 128���� Ư������ �����Ѵ�.
	* ./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
* ������ csv ������ ȣ��Ʈ�� �����Ѵ�.
	* sudo docker cp 9841e190124b:/root/openface/generated-embeddings ~/image_augmentation/

Host���� �н��� �����Ѵ�.
* ���� csv file�� �д´�.
```
df = pd.read_csv("/home/banana/image_augmentation/reps.csv")
X = df.iloc[:,:].values
df_2 = pd.read_csv("/home/banana/image_augmentation/labels.csv")
Y = df_2.iloc[:,0].values
```

* Training Data �� Testing Data�� �и��ϰ� CNN �н��� ���� 16*8 ���·� reshape ���ش�.
```
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)
X_train = X_train.reshape(-1,16,8,1)
X_test = X_test.reshape(-1,16,8,1)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
```

* �н��� �����Ѵ�.
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