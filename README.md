# 194-Face-Point-Detection-CNN
This Model will detect 194 points on the face and has scored a loss = 0.05 and val_loss=50 on training and testing images. The model gives best results to 3:4 WIDTH:HEIGHT as most images in the dataset when were 3:4. The model is not perfect, but poses of front facing face's eyebrows, eyes and nose are perfect. 

# Challenges and Obstacles
The main challenges and obstacles were the different poses of the head and too many points. The final Dense layer was 388 making it harder for the model to detect so many points at the same time simulataneously for so many poses.

# Improvements needed
Reducing the number of points specially the jaw, mouth, nose and eyes. The jaw and mouth have a lot of errors. Improvments in annodation data is very much need due to less variety of data.

# Model
model = tf.keras.models.Sequential([
    tf.compat.v1.keras.layers.Conv2D(16, (3, 3), input_shape=(256, 256, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.3),
    
    tf.compat.v1.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.4),
    
    tf.compat.v1.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.5),
    
    tf.compat.v1.keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.5),
    
    tf.compat.v1.keras.layers.Conv2D(256, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)), 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(256, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(388)
])

I have added Dropout layers and Reguralized the model to reduce overfitting and underfitting. I had Early Stopping in place, but it did not work well.

# Results
![Result1](https://github.com/NiksanJP/194-Face-Point-Detection-CNN/blob/master/results/1.jpg) ![Result2](https://github.com/NiksanJP/194-Face-Point-Detection-CNN/blob/master/results/2.jpg) ![Result3](https://github.com/NiksanJP/194-Face-Point-Detection-CNN/blob/master/results/3.jpg) ![Result4](https://github.com/NiksanJP/194-Face-Point-Detection-CNN/blob/master/results/4.jpg)
