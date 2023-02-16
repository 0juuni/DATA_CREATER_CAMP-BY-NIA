import tensorflow as tf
import numpy as np

from keras import backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# 경로는 바꿔줘야 됩니다. (상대경로로 입력시 이상하게 오류가 뜹니다.)
# 모델 체크포인트 파일은 .pth, .ckpt 형태가 아니라 폴더형태입니다.
path_to_model = "C:/Users/ADMIN/OneDrive - 대전대신고등학교 (1)/바탕 화면/[고등부] Finder 최종 제출 서류/model_checkpoint"
model = tf.keras.models.load_model(path_to_model, custom_objects={'get_f1':get_f1})

test_img = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/ADMIN/OneDrive - 대전대신고등학교 (1)/바탕 화면/[고등부] Finder 최종 제출 서류/test_example',
    # 경로는 바꿔줘야 됩니다. (상대경로로 입력시 이상하게 오류가 뜹니다.)
    # test_data는 전처리를 수행한 후 넣어주셔야합니다. 
    # (문의하니 test셋을 위한 전처리 코드를 따로 정리해 제출할 필요는 없다고 해서 따로 제출하지 않았습니다.)
    label_mode	= 'categorical',
    seed = 18,
    image_size = (128, 128),
    batch_size = 32,
    color_mode = 'grayscale',
    subset = None, 
    validation_split= None
)

def pre(i, j):
  i = tf.cast(i/255.0, tf.float32)
  return i, j

test_img = test_img.map(pre)

test_loss, test_acc, f1_score = model.evaluate(test_img)
print(f1_score)