from sklearn.model_selection import train_test_split



indianDF = pd.read_csv('pima-indians-diabetes.data.csv')
indianDF

arr = indianDF.values
arr
x_data = arr[:,:-1]
y_data = arr[:,[-1]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, stratify= y_data, random_state= 1) ## x데이터를 7:3 으로 랜덤하게 분할해준다.

IOP = Dense( units=1 , input_dim=8, activation='sigmoid') # w, b초기값..
model = Sequential( [IOP] )
model.compile( loss='binary_crossentropy', optimizer= Adam(0.1), metrics= ['acc'] ) #코스함수, 옵티마이저
#h = model.fit( x_dataN, y_data, epochs=500 )
h = model.fit( x_train, y_train, epochs=500 )

p = model.predict_classes(x_test)

(p == y_test).mean() #0.6493506493506493의 정확도를 가짐
 


## 위 두줄 대신 사용가능함
# model.evaluate(x_test, y_test)
# >> 출력 : [2.5337607860565186, 0.649350643157959]
# model.evaluate(x_train, y_train)
# >> 출력 : [2.2660672664642334, 0.6778398752212524]
## 첫 번째는 코스트 두 번째는 정확도