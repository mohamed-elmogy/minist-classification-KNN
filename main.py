from keras.datasets import mnist
import KNN
import Proccessing

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]
y_train = y_train[:10000]
x_train, x_test = Proccessing.pic_divider(x_train, x_test)
train_features = Proccessing.center_calc(x_train)
test_features = Proccessing.center_calc(x_test)
y_train = Proccessing.target_processing(y_train)
model = KNN(train_features, y_train, 3)
predictions = model.predict(test_features)
print(Proccessing.acc(y_test, predictions))
