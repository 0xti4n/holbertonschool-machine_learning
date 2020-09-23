#!/usr/bin/env python3
"""Transfer Knowledge"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """pre-processes the model data"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == "__main__":
    filepath = 'cifar10.h5'
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    base_model = K.applications.ResNet50(include_top=False,
                                         weights="imagenet",
                                         input_shape=(224, 224, 3))

    model = K.models.Sequential()
    model.add(K.layers.UpSampling2D((7, 7)))
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=5,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        verbose=1)

    model.save(filepath)
