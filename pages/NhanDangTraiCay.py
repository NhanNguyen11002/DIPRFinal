import streamlit as st
import cv2 
import joblib


st.sidebar.markdown("# Nhan Dang Trai Cay (coc, oi) ❄️")


st.title('Nhan Dang Trai Cay (coc, oi)')

ftypes = ['jpg','tif','bmp', 'gif', 'png']   


import sys
if 'gelu' in sys.modules:
    delattr(sys.modules['gelu'], 'gelu')
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import cv2 
import numpy as np

import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
#tf.compat.v1.keras.backend.set_session
#print("Tran Tien Duc - Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#CONFIG_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/pipeline.config'
CONFIG_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'

#CHECKPOINT_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/'
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/'

ANNOTATION_PATH = 'Tensorflow/workspace/annotations'

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

def XoaTrung(a, L):
    index = []
    flag = np.zeros(L, bool) ##np.bool
    for i in range(0, L):
        if flag[i] == False:
            flag[i] = True
            x1 = (a[i,0] + a[i,2])/2
            y1 = (a[i,1] + a[i,3])/2
            for j in range(i+1, L):
                x2 = (a[j,0] + a[j,2])/2
                y2 = (a[j,1] + a[j,3])/2
                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if d < 0.2:
                    flag[j] = True
            index.append(i)
    for i in range(0, L):
        if i not in index:
            flag[i] = False
    return flag

def onRecognition(imgin):
    image_np = np.array(imgin)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
            for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    my_box = detections['detection_boxes']
    my_class = detections['detection_classes']+label_id_offset
    my_score = detections['detection_scores']

    my_score = my_score[my_score >= 0.7]
    L = len(my_score)
    my_box = my_box[0:L]
    my_class = my_class[0:L]
    
    flagTrung = XoaTrung(my_box, L)
    my_box = my_box[flagTrung]
    my_class = my_class[flagTrung]
    my_score = my_score[flagTrung]

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes']+label_id_offset,
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=5,
    #         min_score_thresh=.5,
    #         agnostic_mode=False)

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            my_box,
            my_class,
            my_score,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)
    return image_np_with_detections


if __name__ == '__main__':
    uploaded_file = st.file_uploader("Chọn hình")
    if uploaded_file is not None:
        image_path = 'C:/Nhan/Nam3/Ki2/Xulianh/cuoiki/XuLyAnh/testimages/'
        st.image(uploaded_file)
        image_path += uploaded_file.name
        global imgin
        imgin = cv2.imread(image_path,cv2.IMREAD_COLOR)
        imgout = onRecognition(imgin)
        st.write('Hình ảnh sau nhận dạng:')
        st.image(imgout)