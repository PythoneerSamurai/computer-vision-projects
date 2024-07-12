'''
Note:
The labels are as follows:
1. Benign
2. [Malignant] Pre-B
3. [Malignant] Pro-B
4. [Malignant] early Pre-B
'''

# imports
import keras

# data loading
DATA = keras.utils.image_dataset_from_directory("/kaggle/input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]", image_size=(128, 128), batch_size=64)
TRAIN_DATA = DATA.take(int(len(DATA)*0.8))
VAL_DATA = DATA.take(int(len(DATA)*0.2))

# model implementation
MODEL = keras.Sequential([
    keras.layers.Rescaling(scale=1./255),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=keras.activations.relu),
    keras.layers.Dense(128, activation=keras.activations.relu),
    keras.layers.Dense(32, activation=keras.activations.relu),
    keras.layers.Dense(4, activation=keras.activations.softmax),
])
'''MODEL.summary()'''

# callbacks initialization
EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=5)
MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(filepath="bloodCancer.keras", monitor="val_loss", save_best_only=True, mode="min", save_freq="epoch")
CALLBACKS = [EARLY_STOPPING, MODEL_CHECKPOINT]

# parameters
EPOCHS = 20

# compilation, training, saving
MODEL.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
results = MODEL.fit(TRAIN_DATA, epochs=EPOCHS, validation_data=VAL_DATA, callbacks=CALLBACKS)
