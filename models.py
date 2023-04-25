from keras.models import Sequential
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
import time
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def save_list(my_list, file_name):
    df = pd.DataFrame(my_list, columns=['my_column'])
    df.to_csv(f'{file_name}', index=False)

filters=32
# classes= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def define_model(n_layers):
    model = Sequential()
    for i in range(n_layers):
        model.add(Conv2D(filters*(2**i), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',  input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.001, momentum=0.9)
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

models=[]
times=[]
history=[]
test_accuracy=[]
train_accuracy=[]
for i in range(5):
    model_=define_model(i+1)
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    it_train = datagen.flow(x_train, y_train, batch_size=64)
    steps = int(x_train.shape[0] / 64)
    start=time.time()
    history_ = model_.fit(it_train, steps_per_epoch=steps, epochs=20, validation_data=(x_test, y_test))
    end=time.time()
    # start=time.time()
    # history_ = model_.fit(x_train, y_train,batch_size=64,epochs=20,validation_data=(x_test, y_test),shuffle=True)
    # end=time.time()
    models.append(model_)
    times.append(end-start)
    history.append(history_.history['loss'][-1])
    test_loss, test_acc = model_.evaluate(x_test, y_test, verbose=1)
    train_loss, train_acc = model_.evaluate(x_train, y_train, verbose=1)
    test_accuracy.append(test_acc)
    train_accuracy.append(train_acc)

for i,model_ in enumerate(models):
    model_.save(f"model{i+1}.h5")

    
save_list(times, "times.csv")
save_list(history, "history.csv")
save_list(test_accuracy, "test_accuracy.csv")
save_list(train_accuracy, "train_accuracy.csv")

def ret():
    return models, times, history, test_accuracy, train_accuracy




