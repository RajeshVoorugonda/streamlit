import streamlit as st
import io
from contextlib import redirect_stdout
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

#Install required packages for streamlit
#import os

# Install packages listed in requirements.txt
#os.system('pip install -r requirements.txt')
def load_saved(file_name):
    df = pd.read_csv(f'{file_name}')
    loaded_list = df['my_column'].tolist()
    return loaded_list


times=load_saved("times.csv")
history=load_saved("history.csv")
test_accuracy=load_saved("test_accuracy.csv")
train_accuracy=load_saved("train_accuracy.csv")

def each_model():

    st.title("Convolutional Neural Network")



    n_layers = st.slider('Number of Convolutional Layers', 1, 5, 1)

    model=load_model(f'model{n_layers}.h5')
    st.write('Test Accuracy:', test_accuracy[n_layers-1])
    st.write('Train Accuracy:', train_accuracy[n_layers-1])


    no_of_params=model.count_params()
    st.write('Number of Parameters:', no_of_params)


    st.write('Training loss: ', history[n_layers-1])

    st.write('Training Time:', times[n_layers-1])



    buffer = io.StringIO()

    with redirect_stdout(buffer):
        model.summary()
        
    summary_string = buffer.getvalue()
    summary_list = summary_string.strip().split('\n')

    layer_info = []
    for row in summary_list[1:]:
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

    layer_info=layer_info[1:]
    df = pd.DataFrame(layer_info)
    st.header("Model Details")
    st.write(df)



def compare_plots():
    st.title("Plots comparing the models")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    k=[1,2,3,4,5]
    axs[1,0].plot(k,history)
    axs[1,0].set_ylim(0.5, 1.5)
    axs[1,0].set_title('Training loss Vs No.of layers')
    axs[0,0].plot(k,test_accuracy)
    axs[0,0].set_ylim(0.6, 0.8)
    axs[0,0].set_title('Test accuracy Vs No.of layers')
    axs[0,1].plot(k,train_accuracy)
    axs[0,1].set_ylim(0.6, 0.8)
    axs[0,1].set_title('Training accuracy Vs No.of layers')
    axs[1,1].plot(k,times)
    axs[1,1].set_ylim(3000, 5000)
    axs[1,1].set_title('Training Time Vs No.of layers')

    st.pyplot(fig)

with st.sidebar:
    st.write("# Select")

menu = ["Models", "Compare Models"]
choice = st.sidebar.selectbox("", menu)


    

if choice == "Models":
    each_model()
elif choice == "Compare Models":
    compare_plots()


    


