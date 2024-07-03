# imports
import keras

# data loading and handling
DATA = keras.utils.image_dataset_from_directory('/kaggle/input/ucf-dataset/Train', image_size=(64, 64))
TRAIN_DATA = DATA.take(int(len(DATA)*0.7))
VALIDATION_DATA = DATA.take(int(len(DATA)*0.3))

# model implementation
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
    keras.layers.Flatten(),
    keras.layers.Dense(128),
    keras.layers.Dense(16),
    keras.layers.Dense(14, activation="softmax"),
])

MODEL.summary()

# compilation, training, saving
MODEL.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
results = MODEL.fit(TRAIN_DATA, epochs=50, validation_data=VALIDATION_DATA)
MODEL.save("ucf_model.keras")
