import streamlit as st
import numpy as np

model = np.load('weight.npz')
x_mean = model['x_mean']
x_std = model['x_std']
theta = model['theta']

@st.cache
def predict(carat, cut, color, clarity, depth, table, x, y, z, x_mean, x_std, theta):
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    cut = cut_mapping.get(cut, 0)
    color = color_mapping.get(color, 0)
    clarity = clarity_mapping.get(clarity, 0)

    input = np.array([carat, depth, table, x, y, z, cut, color, clarity], dtype='float')
    
 
    input = (input - x_mean) / x_std

  
    input = np.concatenate(([1.0], input)) 
    

    prediction = input.dot(theta)
    return prediction

st.title('DIAMOND PRICES PREDICTION')

st.header('Vui lòng hãy nhập thông tin viên kim cương bạn muốn mua:')
carat = st.number_input('Carat Weight: ', min_value=0.1, max_value=10.0, value=1.0)
cut = st.selectbox('Cut Rating: ', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color Rating: ', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity Rating: ', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Diamond Depth Percentage: ', min_value=0.1, max_value=100.0, value=1.0)
table = st.number_input('Diamond Table Percentage: ', min_value=0.1, max_value=100.0, value=1.0)
x = st.number_input('Diamond Length (X) in mm: ', min_value=0.1, max_value=100.0, value=1.0)
y = st.number_input('Diamond Width (Y) in mm: ', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('Diamond Depth (Z) in mm: ', min_value=0.1, max_value=100.0, value=1.0)

if st.button('Predict Price'):
    try:
        out = predict(carat, cut, color, clarity, depth, table, x, y, z, x_mean, x_std, theta)
        if np.issubdtype(out.dtype, np.number):
            st.success(f"Giá dự đoán của viên kim cương là: ${out[0]:.2f} USD")
        else:
            st.error(f"Đã xảy ra lỗi khi dự đoán giá: {out}")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")

