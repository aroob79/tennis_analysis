from keyPointExtractorModel import keyPointModel
import numpy as np
import os
import cv2
import json
import pickle
from matplotlib import pyplot as plt
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')


# first extract the input data and output point


def load_and_preprocess_data(output_file_path, mode, data_path="None", img_folder_path=None, output_shape=(224, 224)):

    save_img_path = os.path.join(output_file_path, mode+'_saved_images.pkl')
    save_point_path = os.path.join(
        output_file_path, mode+'_saved_points_norm.pkl')

    if os.path.exists(save_img_path) and os.path.exists(save_point_path):
        with open(save_img_path, 'rb') as file:
            images = pickle.load(file)
        file.close()
        with open(save_point_path, 'rb') as file:
            points = pickle.load(file)
        file.close()

        points[:, :, 0] *= output_shape[0]
        points[:, :, 1] *= output_shape[1]

        return np.array(images), points
    else:
        # load the data/keypoint from json file
        with open(data_path, 'rb') as file:
            dt = json.load(file)
        file.close()

        # load the corresponding inage of the point and normalize the point
        images = []
        points = []
        for value in dt:
            name_part = value['id']
            img_name = os.path.join(img_folder_path, name_part)
            img_name = img_name + '.png'
            img = cv2.imread(img_name)
            h, w, _ = img.shape
            # resize the image to 224 by 224
            img = cv2.resize(img, output_shape)
            point = np.array(value['kps']).astype(np.float32)
            point[:, 0] = point[:, 0]/w
            point[:, 1] = point[:, 1]/h
            points.append(point.tolist())
            images.append(img)
        points = np.array(points)
        # store the into pickle file
        with open(save_img_path, 'wb') as file:
            pickle.dump(images, file)
        file.close()
        with open(save_point_path, 'wb') as file:
            pickle.dump(points, file)
        file.close()
        points[:, :, 0] *= output_shape[0]
        points[:, :, 1] *= output_shape[1]

        return np.array(images), points


# find the training data
json_path = r'E:\python\basic_code\tennis_analysis_using_yolo\keypoint_dataset\data\data_train.json'
img_path = r'E:\python\basic_code\tennis_analysis_using_yolo\keypoint_dataset\data\images'
output_file_path = r'E: \python\basic_code\tennis_analysis_using_yolo'


train_images, train_points = load_and_preprocess_data(
    output_file_path=output_file_path, mode='train', output_shape=(224, 224))


# find the test data
json_path = r'E:\python\basic_code\tennis_analysis_using_yolo\keypoint_dataset\data\data_val.json'
img_path = r'E:\python\basic_code\tennis_analysis_using_yolo\keypoint_dataset\data\images'


test_images, test_points = load_and_preprocess_data(
    output_file_path=output_file_path, mode='test', output_shape=(224, 224))


model = keyPointModel()
# trainig the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.MeanSquaredError()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# define the model check point
model_cheek_point = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

summary = model.fit(train_images, train_points, epochs=50, validation_data=(test_images, test_points), callbacks=[
                    model_cheek_point, early_stopping])


model.save_weights('model_weights.keras')
if __name__ == '__main__':
    pass
