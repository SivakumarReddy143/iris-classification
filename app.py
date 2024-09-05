import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import numpy as np
import os

model=pickle.load(open('../model/model.pkl','rb'))

sepal_length=float(st.text_input('Enter sepal_length:'))
sepal_width=float(st.text_input('Enter sepal_width:'))
petal_length=float(st.text_input('Enter petal_length:'))
petal_width=float(st.text_input('Enter petal_width:'))
array=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
scaler=StandardScaler()
test_array=scaler.fit_transform(array)
result_list=['setosa', 'versicolor', 'virginica']
button=st.button('predict')
flowers=os.listdir('../images')
if button:
    response=model.predict(test_array)[0]
    result=result_list[response]
    st.write(result)
    st.image(f"../images/{flowers[response]}",width=500,use_column_width=500)

