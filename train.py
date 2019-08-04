import tensorflow as tf 
import numpy as np 
import os 
import gc
import cv2

model = tf.keras.models.Sequential([
    tf.compat.v1.keras.layers.Conv2D(16, (3, 3), input_shape=(256, 256, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    #tf.keras.layers.Dropout(0.3),
    
    tf.compat.v1.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    #tf.keras.layers.Dropout(0.4),
    
    tf.compat.v1.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    #tf.keras.layers.Dropout(0.5),
    
    tf.compat.v1.keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    #tf.keras.layers.Dropout(0.5),
    
    tf.compat.v1.keras.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)), 
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(256, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(388)
])

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
print(model.summary())

es = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')

X = np.load('X2.npy')
Y = np.load('Y2.npy')

xTest = np.load('Xtest.npy')
yTest = np.load('Ytest.npy')

print(X.shape, Y.shape)

try:
    model.load_weights('weights/weights.h5')
    print("Model Loaded")
except Exception:
    pass
    
while True:
    history = model.fit(X, Y, epochs=100, batch_size=64, validation_data=(xTest, yTest), shuffle=True, callbacks=[es])
    model.save_weights('weights/weights3.h5')