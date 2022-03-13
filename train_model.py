from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

TRAIN_DIR = "./Data/train"
TEST_DIR = "./Data/test"

model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.5),

    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

train_data = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_gen = train_data.flow_from_directory(TRAIN_DIR, batch_size=10, target_size=(150, 150))

test_data = ImageDataGenerator(rescale=1.0/255)
test_gen = test_data.flow_from_directory(TEST_DIR, batch_size=10, target_size=(150, 150))

checkpoint = ModelCheckpoint('model{epoch:02d}.model',monitor='val_loss',save_best_only=True)
history = model.fit(train_gen, epochs=14, validation_data=test_gen, callbacks=[checkpoint])                                                        