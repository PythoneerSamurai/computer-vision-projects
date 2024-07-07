# imports
import tensorflow as tf
import keras

# initializing strategy
strategy = tf.distribute.MirroredStrategy()

# parameters
EPOCHS = 30

# data loading
TRAIN_DATA = keras.utils.image_dataset_from_directory('/kaggle/input/sarscov2-ctscan-dataset', image_size=(128, 128), batch_size=32)

# model implementation
with strategy.scope():
    MODEL = keras.Sequential([
        keras.layers.Rescaling(scale=1./255),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(16, activation=keras.activations.relu),
        keras.layers.Dense(1, activation=keras.activations.sigmoid),
    ])
    '''MODEL.summary()'''

    # callback initialization
    CALLBACK = keras.callbacks.EarlyStopping(
        monitor='loss',
        mode='min',
        patience=5,
    )

    # compilation, training
    MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    results = MODEL.fit(TRAIN_DATA, epochs=EPOCHS, callbacks=[CALLBACK])

# model saving
MODEL.save("sarsCov.keras")
