import streamlit as st
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
import time
import io
from contextlib import redirect_stdout

st.title("Convolutional Neural Network")
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
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


n_layers = st.slider('Number of Convolutional Layers', 1, 5, 1)

model = define_model(n_layers)
start=time.time()
history = model.fit(x_train, y_train,batch_size=32,epochs=1,validation_data=(x_test, y_test),shuffle=True)
end=time.time()


st.write('Training Time:', end-start)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
st.write('Test Accuracy:', test_acc)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=1)
st.write('Train Accuracy:', train_acc)


no_of_params=model.count_params()
st.write('Number of Parameters:', no_of_params)


st.write('Training loss: ', history.history['loss'][-1])

# buffer = io.StringIO()
# with redirect_stdout(buffer):
#     model.summary()
# summary = pd.read_csv(io.StringIO(buffer.getvalue().replace('_________________________________________________________________\n', '')), sep=' ', index_col=0)
# # st.write(buffer.getvalue())
# st.table(summary)


buffer = io.StringIO()

# use redirect_stdout to capture the output of model.summary()
with redirect_stdout(buffer):
    model.summary()

# parse the captured output as a string and split it by newlines
summary_string = buffer.getvalue()
summary_list = summary_string.strip().split('\n')

# create a list of dictionaries with the layer information
layer_info = []
for row in summary_list[1:]:
    # check if it's a separator row (e.g. "===") or a layer row
    if not row.startswith(' '):
        separator = row.strip()
    else:
        layer = row.strip().split()
        if(len(layer)>1):
            if(len(layer)>5):
                layer_name = ' '.join(layer[:-5])
                layer_shape = layer[2:-1]
                layer_params = layer[-1]
                layer_info.append({'Layer': layer_name, 'Output Shape': layer_shape, 'Param #': layer_params})
            else:
                layer_name = ' '.join(layer[:-3])
                layer_shape = layer[2:-1]
                layer_params = layer[-1]
                layer_info.append({'Layer': layer_name, 'Output Shape': layer_shape, 'Param #': layer_params})


# create a pandas dataframe from the layer info
layer_info=layer_info[1:]
df = pd.DataFrame(layer_info)
st.header("Model")
# display the dataframe in a table using st.write()
st.write(df)

model.summary()
