import streamlit as st
import numpy as np
from PIL import Image
import os

# Load the model
model = np.load('weight.npz')
x_mean = model['x_mean']
x_std = model['x_std']
theta = model['theta']

# Ensure the image file exists
image_path = 'cut_image.jpg'
if not os.path.exists(image_path):
    st.error(f"Hình ảnh {image_path} không tìm thấy. Vui lòng kiểm tra đường dẫn.")
else:
    cut_image = Image.open(image_path)

@st.cache_resource
def predict(carat, cut, color, clarity, depth, table, x, y, z, x_mean, x_std, theta):
    # Mappings for categorical features
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

    # Convert categorical features to numeric
    cut = cut_mapping.get(cut, 0)
    color = color_mapping.get(color, 0)
    clarity = clarity_mapping.get(clarity, 0)

    # Prepare the input array
    input = np.array([carat, depth, table, x, y, z, cut, color, clarity], dtype='float')
    
    # Standardize the input
    input = (input - x_mean) / x_std

    # Add a bias term (intercept)
    input = np.concatenate(([1.0], input))

    # Perform prediction using the model
    prediction = input.dot(theta)
    return prediction

# Streamlit UI elements
st.title('Diamond Prices Prediction')

# Display an image for diamond cut ratings if it exists
if os.path.exists(image_path):
    st.header('Hướng dẫn cắt kim cương:')
    st.image(cut_image, caption='Các loại cắt kim cương', use_column_width=True)

st.header('Vui lòng nhập thông tin về viên kim cương bạn muốn mua:')
carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0, step=0.01)
cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Diamond Depth Percentage:', min_value=1.0, max_value=100.0, value=60.0, step=0.1)
table = st.number_input('Diamond Table Percentage:', min_value=1.0, max_value=100.0, value=55.0, step=0.1)
x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=5.0, step=0.1)
y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=5.0, step=0.1)
z = st.number_input('Diamond Depth (Z) in mm:', min_value=0.1, max_value=100.0, value=3.0, step=0.1)

# Prediction button
if st.button('Predict Price'):
    if carat > 0 and depth > 0 and table > 0 and x > 0 and y > 0 and z > 0:
        out = predict(carat, cut, color, clarity, depth, table, x, y, z, x_mean, x_std, theta)
        st.success(f"Giá dự đoán của viên kim cương là: ${out[0]:.2f} USD")
    else:
        st.error("Vui lòng nhập các giá trị hợp lệ cho tất cả các trường.")
