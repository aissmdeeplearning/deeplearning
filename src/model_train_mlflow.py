import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16  # type: ignore
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras

def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']

    if train:
        img_size = config['model']['image_size']
        train_path = config['model']['train_path']
        test_path = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertical_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model'].get('optimizer', 'adam')  # Ensure optimizer has a default value
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['sav_dir']

        print(f"Batch size type: {type(batch)}")  # Debugging batch type
        print(f"Optimizer from config: {optimizer}")  # Debug print

        # Load pre-trained VGG16 model (without top layers)
        resnet = VGG16(input_shape=img_size + [3], weights='imagenet', include_top=False)

        # Freeze pre-trained layers
        for layer in resnet.layers:
            layer.trainable = False

        # Custom classifier on top of VGG16
        x = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(x)
        mod = Model(inputs=resnet.input, outputs=prediction)

        print(mod.summary())

        img_size = tuple(img_size)

        # Convert optimizer string to actual Keras optimizer
        optimizers = {
            "adam": tf.keras.optimizers.Adam(),
            "sgd": tf.keras.optimizers.SGD(),
            "rmsprop": tf.keras.optimizers.RMSprop(),
            "adagrad": tf.keras.optimizers.Adagrad(),
            "adamax": tf.keras.optimizers.Adamax()
        }

        # Ensure optimizer is valid
        if not isinstance(optimizer, str):
            optimizer = "adam"
        
        mod.compile(loss=loss, optimizer=optimizers.get(optimizer.lower(), tf.keras.optimizers.Adam()), metrics=metrics)

        # Data Augmentation
        train_gen = ImageDataGenerator(rescale=rescale, shear_range=shear_range, zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, rotation_range=90)
        test_gen = ImageDataGenerator(rescale=rescale)

        train_set = train_gen.flow_from_directory(train_path, target_size=img_size, batch_size=batch, class_mode=class_mode)
        test_set = test_gen.flow_from_directory(test_path, target_size=img_size, batch_size=batch, class_mode=class_mode)

        # MLFLOW Integration
        mlflow_config = config['mlflow_config']
        remote_server_uri = mlflow_config["remote_server_uri"]
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run():
            history = mod.fit(train_set, epochs=epochs, validation_data=test_set,
                              steps_per_epoch=len(train_set), validation_steps=len(test_set))

            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history.get('val_accuracy', history.history.get('val_acc'))[-1]  # Handle different versions

            mlflow.log_param("epochs", epochs)
            mlflow.log_param("loss", loss)
            mlflow.log_param("val_loss", val_loss)
            mlflow.log_param("val_accuracy", val_acc)
            mlflow.log_param("metrics", val_acc)

            tracking_url_type_Store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_Store != "file":
                mlflow.keras.log_model(mod, "model", registered_model_name=mlflow_config["registered_model_name"])
            else:
                mlflow.keras.log_model(mod, "model")

    else:
        print("Model is not trainable")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_model_mlflow(config_file=passed_args.config)
