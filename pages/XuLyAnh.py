import streamlit as st
import cv2 
import joblib
import chapter03 as c3

st.sidebar.markdown("# Xu Ly Anh ❄️")

app_mode = st.sidebar.selectbox('Select Page',['Negative','Logarit','Power','PiecewiseLinear','Histogram','HistEqual','HistEqualColor','LocalHist'
                                               ,'HistStat','BoxFilter','Smooth','MedianFilter','Sharpen']) 

ftypes = ['jpg','tif','bmp', 'gif', 'png']

if app_mode == 'Negative':
    st.title('Negative')
    
    uploaded_file = st.file_uploader("Chọn hình", key=1)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        global imgin
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Negative(imgin))

elif app_mode == 'Logarit':
    st.title('Logarit')
    uploaded_file = st.file_uploader("Chọn hình", key=2)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Logarit(imgin))
elif app_mode == 'Power':
    st.title('Power')
    uploaded_file = st.file_uploader("Chọn hình", key=3)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Power(imgin))
elif app_mode == 'PiecewiseLinear':
    st.title('PiecewiseLinear')
    uploaded_file = st.file_uploader("Chọn hình", key=4)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.PiecewiseLinear(imgin))
elif app_mode == 'Histogram':
    st.title('Histogram')
    uploaded_file = st.file_uploader("Chọn hình", key=5)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Histogram(imgin))
elif app_mode == 'HistEqual':
    st.title('HistEqual')
    uploaded_file = st.file_uploader("Chọn hình", key=6)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.HistEqual(imgin))
elif app_mode == 'HistEqualColor':
    st.title('HistEqualColor')
    uploaded_file = st.file_uploader("Chọn hình", key=7)
    if uploaded_file is not None:
        image_path = 'D:/Nam3/XLA/TaiLieu_XuLyAnh_ThiGiacMay/SachXuLyAnh/DIP3E_Original_Images_CH06/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_COLOR)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.HistEqualColor(imgin))
elif app_mode == 'LocalHist':
    st.title('LocalHist')
    uploaded_file = st.file_uploader("Chọn hình", key=8)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.LocalHist(imgin))
elif app_mode == 'HistStat':
    st.title('HistStat')
    uploaded_file = st.file_uploader("Chọn hình", key=9)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.HistStat(imgin))
elif app_mode == 'BoxFilter':
    st.title('BoxFilter')
    uploaded_file = st.file_uploader("Chọn hình", key=10)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.BoxFilter(imgin))
elif app_mode == 'Smooth':
    st.title('Smooth')
    uploaded_file = st.file_uploader("Chọn hình", key=11)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Smooth(imgin))
elif app_mode == 'MedianFilter':
    st.title('MedianFilter')
    uploaded_file = st.file_uploader("Chọn hình", key=12)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.MedianFilter(imgin))
elif app_mode == 'Sharpen':
    st.title('Sharpen')
    uploaded_file = st.file_uploader("Chọn hình", key=13)
    if uploaded_file is not None:
        image_path = 'testimages/Chuong3/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        imgin = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        st.write('Hình ảnh sau khi xử lý:')
        st.image(c3.Sharpen(imgin))



    

