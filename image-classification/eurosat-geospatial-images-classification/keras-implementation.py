# imports
import keras

# data loading
DATA = keras.utils.image_dataset_from_directory("/kaggle/input/eurosat-dataset/EuroSAT", image_size=(64, 64), batch_size=64)
TRAIN_DATA = DATA.take(int(len(DATA)*0.8))
VAL_DATA = DATA.take(int(len(DATA)*0.2))

# model implementation
MODEL = keras.Sequential([
    keras.layers.Rescaling(scale=1./255),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(512),
    keras.layers.Dense(128),
    keras.layers.Dense(32),
    keras.layers.Dense(10, activation=keras.activations.softmax),
])
'''MODEL.summary()'''

# callbacks initialization
EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=7)
MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(filepath="eurosat.keras", monitor="loss", save_best_only=True, mode="min", save_freq="epoch")
CALLBACKS = [EARLY_STOPPING, MODEL_CHECKPOINT]

# parameters
EPOCHS = 30

# compilation, training, saving
MODEL.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
results = MODEL.fit(TRAIN_DATA, epochs=EPOCHS, validation_data=VAL_DATA, callbacks=CALLBACKS)
