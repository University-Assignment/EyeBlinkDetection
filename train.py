import datetime
import numpy as np
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

xTrain = np.load('dataset/xTrain.npy').astype(np.float32)
yTrain = np.load('dataset/yTrain.npy').astype(np.float32)
xTest = np.load('dataset/xTest.npy').astype(np.float32)
yTest = np.load('dataset/yTest.npy').astype(np.float32)

print(xTrain.shape, yTrain.shape)
print(xTest.shape, yTest.shape)

trainDatagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

testDatagen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDatagen.flow(
    x=xTrain, y=yTrain,
    batch_size=32,
    shuffle=True
)

testGenerator = testDatagen.flow(
    x=xTest, y=yTest,
    batch_size=32,
    shuffle=False
)


inputs = Input(shape=(24, 24, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit_generator(
    trainGenerator, epochs=50, validation_data=testGenerator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ]
)

model = load_model('models/%s.h5' % (start_time))

y_pred = model.predict(xTest/255.)
y_pred_logical = (y_pred > 0.5).astype(np.int)

print ('test acc: %s' % accuracy_score(yTest, y_pred_logical))