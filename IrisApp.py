from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st


def title():
    # Declare title app
    #img = Image.open("Naresuan_University_Logo.png")
    #st.image(img, width=200)
    #st.title("Iris Classification Web App")

    title_container = st.container()
    col1, col2 = st.columns([20,1])
    image = Image.open("image_logo.png")
    with title_container:
        with col1:
            st.markdown('<h1 style="color: purple;">Iris Classification Web App</h1>',
                            unsafe_allow_html=True)
        with col2:
            st.image(image, width=200)


    st.write("""Iris is a flower family that includes several species such as
setosa, versicolor, virginica, and others.""")
    #st.subheader('Demo for image classification.')
    #img = Image.open("versicolor.png")
    #st.image(img, width=500)
    col1, col2 = st.columns(2)
    original = Image.open("fig1.png")
    col1.subheader("Iris setosa")
    col1.image(original, use_column_width=True)

    grayscale = Image.open("Iris-Setosa1.jpg")
    col2.subheader("Iris versicolor")
    col2.image(grayscale, use_column_width=True)

    st.write("The iris dataset description.")

    st.write(
    pd.DataFrame({
      'Sepal length': [5.1, 4.9,4.4, 6.3, 6.1],
      'Sepal width': [3.5, 3.0,2.9, 2.5, 2.8],
      'Petal length': [1.4, 1.4,1.5, 4.9, 4.7],
      'Petal width': [0.2, 0.3,0.4, 1.5, 1.2],
      'Class': ['Iris Setosa', 'Iris Setosa', 'Iris Setosa', 'Iris Versicolour', 'Iris Versicolour']
        }))
    #st.success("Done!")
def predict(values):
    w_b = pd.read_csv('w_b_iris.csv').values[:,-1]
    p = np.sign(np.matmul(values,w_b[:-1])+w_b[-1])
    return p

def main():
    # set title
    title()

    # enable users to upload images for the model to make predictions
    #file_up = st.file_uploader("Upload an image", type=['png','jpeg','jpg'])

    with st.sidebar:
        
        st.subheader('Iris data classification')
        number1 = st.number_input('Select sepal length (cm)', min_value=3.5, max_value=7.0, value=4.0, step=0.1)
        number2 = st.number_input('Select sepal wigth (cm)', min_value=2.0, max_value=4.0, value=2.4, step=0.1)
        number3 = st.number_input('Select petal length (cm)', min_value=1.0, max_value=7.0, value=4.0, step=0.1)
        number4 = st.number_input('Select petal wigth (cm)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        values = [number1, number2, number3, number4]
        
        #st.write('The current number is ', values)
        # prediction step
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sepal length (cm):", number1)
    col2.metric("Sepal wigth (cm):", number2)
    col3.metric("Petal length (cm):",number3)
    col4.metric("Petal wigth (cm):",number4)
    p = predict(values) # predict the class by using the CNN feature of the img.
    if p >= 0:
        st.write("""**Prediction**: _Iris setosa_""")
        img = Image.open("Iris-Setosa1.jpg")
        st.image(img, width=200)
        
    else:
        st.write("""**Prediction**: _Iris versicolor_""")
        img = Image.open("Iris_versicolor.jpg")
        st.image(img, width=200)
        


if __name__=="__main__":
    main()