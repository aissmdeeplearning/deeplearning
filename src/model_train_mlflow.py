import numpy as np
import argparse
import mlflow
import mlflow.keras
import tensorflow as tf
from urllib.parse import urlparse
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16
from get_data import get_data


# Function to initialize optimizer safely
def get_optimizer(optimizer_name):
    optimizers = {
        "adam": tf.keras.optimizers.Adam(),
        "sgd": tf.keras.optimizers.SGD(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        "adamax": tf.keras.optimizers.Adamax(),
        "nadam": tf.keras.optimizers.Nadam()
    }
    
    # Ensure optimizer_name is not None before calling .lower()
    if not optimizer_name:
        print("⚠️ Warning: No optimizer specified. Using default 'Adam'.")
        return tf.keras.optimizers.Adam()
    
    return optimizers.get(optimizer_name.lower(), tf.keras.optimizers.Adam())  # Default to Adam


def train_model_mlflow(config_file):
    config = get_data(config_file)

    # Ensure 'trainable' is present
    train = config['model'].get('trainable', True)
    if not train:
        print("❌ Model is not trainable. Exiting...")
        return

    img_size = config['model']['image_size']
    train_set_path = config['model']['train_path']
    test_set_path = config['model']['test_path']
    num_cls = config['load_data']['num_classes']

    # Image augmentation settings
    img_augment = config.get('img_augment', {})
    rescale = img_augment.get('rescale', 1.0)
    shear_range = img_augment.get('shear_range', 0.0)
    zoom_range = img_augment.get('zoom_range', 0.0)
    horizontal_flip = img_augment.get('horizontal_flip', False)
    vertical_flip = img_augment.get('vertical_flip', False)
    class_mode = img_augment.get('class_mode', 'categorical')
    batch = img_augment.get('batch_size', 32)

    # Model settings
    loss = config['model'].get('loss', 'categorical_crossentropy')
    optimizer_name = config['model'].get('optimizer', 'adam')
    metrics = config['model'].get('metrics', ['accuracy'])
    epochs = config['model'].get('epochs', 10)

    # Convert optimizer name to actual optimizer object
    optimizer = get_optimizer(optimizer_name)

    # Define Model
    base_model = VGG16(input_shape=img_size + [3], weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    prediction = Dense(num_cls, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    print(model.summary())

    # Compile Model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Data generators
    train_gen = ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotation_range=90
    )

    test_gen = ImageDataGenerator(rescale=rescale)

    train_set = train_gen.flow_from_directory(
        train_set_path,
        target_size=tuple(img_size),
        batch_size=batch,
        class_mode=class_mode
    )

    test_set = test_gen.flow_from_directory(
        test_set_path,
        target_size=tuple(img_size),
        batch_size=batch,
        class_mode=class_mode
    )

    ################# START OF MLFLOW #################

    mlflow_config = config.get('mlflow_config', {})
    remote_server_uri = mlflow_config.get("remote_server_uri", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config.get("experiment_name", "default_experiment"))

    with mlflow.start_run():
        history = model.fit(
            train_set,
            epochs=epochs,
            validation_data=test_set,
            steps_per_epoch=len(train_set),
            validation_steps=len(test_set)
        )

        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("loss", loss)
        mlflow.log_param("val_loss", val_loss)
        mlflow.log_param("val_accuracy", val_acc)
        mlflow.log_param("metrics", val_acc)

        tracking_url_type_Store = urlparse(mlflow.get_artifact_uri()).scheme

        mlflow.keras.log_model(
            model, "model",
            registered_model_name=mlflow_config.get("registered_model_name", None)
            if tracking_url_type_Store != "file" else None
        )

    print("✅ Training complete!")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_model_mlflow(config_file=passed_args.config)