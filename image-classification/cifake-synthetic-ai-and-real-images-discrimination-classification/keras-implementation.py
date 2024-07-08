# imports
import keras

# data loading and handling
TRAIN_DATA = keras.utils.image_dataset_from_directory('/kaggle/input/cifake-real-and-ai-generated-synthetic-images/train', image_size=(32, 32))
VAL_DATA = keras.utils.image_dataset_from_directory('/kaggle/input/cifake-real-and-ai-generated-synthetic-images/test', image_size=(32, 32))

# model implementation
MODEL = keras.Sequential([
    keras.layers.Rescaling(scale=1./255),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128),
    keras.layers.Dense(16),
    keras.layers.Dense(1, activation=keras.activations.sigmoid),
])

# callbacks initialization
EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=7)
MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(filepath="cifake.keras", monitor="loss", save_best_only=True, mode="min", save_freq="epoch")
CALLBACKS = [EARLY_STOPPING, MODEL_CHECKPOINT]

# parameters
EPOCHS = 30

# compilation, training, saving
MODEL.compile(optimizer='Adam', loss="binary_crossentropy", metrics=['accuracy'])
results = MODEL.fit(TRAIN_DATA, epochs=EPOCHS, validation_data=VAL_DATA, callbacks=CALLBACKS)
