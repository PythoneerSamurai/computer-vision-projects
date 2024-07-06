# imports
import keras

# data loading
TRAIN_DATA = keras.utils.image_dataset_from_directory('/kaggle/input/bf-dataset/bf_dataset/train', image_size=(128, 128), batch_size=32)
VAL_DATA = keras.utils.image_dataset_from_directory('/kaggle/input/bf-dataset/bf_dataset/val', image_size=(128, 128), batch_size=32)

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
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128),
    keras.layers.Dense(16),
    keras.layers.Dense(1, activation=keras.activations.sigmoid),
])
'''MODEL.summary()'''

# callback initialization
CALLBACK = keras.callbacks.EarlyStopping(
    monitor='loss',
    mode='min',
    patience=5,
)

# parameters
EPOCHS = 30

# training, saving
MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results = MODEL.fit(TRAIN_DATA, epochs=EPOCHS, callbacks=[CALLBACK], validation_data = VAL_DATA)
MODEL.save("boneFrac.keras")
