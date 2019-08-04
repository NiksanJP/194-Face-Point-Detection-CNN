import tensorflow as tf 
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

model.load_weights('weights/weights.h5')
print("Model Loaded")


def guessImage(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (256, 256))
    orgImage = img
    
    prediction = model.predict(np.array([img]))
    
    print(prediction)
    
    plt.scatter(x=prediction[0,0::2], y=prediction[0,1::2], c='r', s=1)
    plt.imshow(orgImage)
    plt.waitforbuttonpress()
    
guessImage('testImages/3.jpg')