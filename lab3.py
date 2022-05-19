import sys
sys.stdout = open('output.txt', 'w')
import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from keras import regularizers
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
with tf.device('/device:GPU:0'):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype(np.float32)/255.0
    x_test = x_test.astype(np.float32)/255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)

    batch_size = 128
    n_epoch = 50
    k = 5
    num_model = 0

    def cross_validation(data_gen, dropout_rate, l2_reg):
        global num_model
        accuracy = []
        for train_index, val_index in KFold(k).split(x_train):
            xtrain, xval = x_train[train_index], x_train[val_index]
            ytrain, yval = y_train[train_index], y_train[val_index]

            cnn = Sequential()
            cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
            cnn.add(Conv2D(128, (3, 3), activation='relu'))
            cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            cnn.add(Dropout(dropout_rate[0]))

            cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
            cnn.add(Conv2D(256, (3, 3), activation='relu'))
            cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            cnn.add(Dropout(dropout_rate[1]))

            cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
            cnn.add(Conv2D(512, (3, 3), activation='relu'))
            cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            cnn.add(Dropout(dropout_rate[2]))

            cnn.add(Flatten())

            cnn.add(Dense(1000, activation='relu'))
            cnn.add(Dropout(dropout_rate[3]))

            cnn.add(Dense(1000, activation='relu'))
            cnn.add(Dropout(dropout_rate[4]))

            cnn.add(Dense(100, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))

            cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            cnn.summary()

            if data_gen:
                generator = ImageDataGenerator(rotation_range=3.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
                cnn.fit_generator(generator.flow(xtrain, ytrain, batch_size=batch_size), epochs=n_epoch, validation_data=(x_test, y_test),
                        verbose=2, callbacks=[early_stop])
            else:
                cnn.fit(xtrain, ytrain, batch_size=batch_size, epochs=n_epoch, validation_data=(x_test, y_test), verbose=2,
                        callbacks=[early_stop])
            accuracy.append(cnn.evaluate(xval, yval, verbose=0)[1])
            if num_model == 0:
                cnn.save("my_cnn.h5")
        num_model += 1

        return accuracy

    acc_001 = cross_validation(True, [0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
    acc_010 = cross_validation(True, [0.2, 0.5, 0.5, 0.5, 0.5], 0.0)
    acc_011 = cross_validation(True, [0.2, 0.5, 0.5, 0.5, 0.5], 0.01)

    print("출력형식 : [Data augmentation-Dropout-l2 reg] (교차검증 시도/평균)")
    print("[001] (", acc_001, "/", np.array(acc_001).mean(), ")")
    print("[010] (", acc_010, "/", np.array(acc_010).mean(), ")")
    print("[011] (", acc_011, "/", np.array(acc_011).mean(), ")")

    import matplotlib.pyplot as plt

    plt.grid()
    plt.boxplot([acc_001, acc_010, acc_011]
                , labels = ["001", "010", "011"])
    plt.show()
    plt.savefig('myfigure.png', dpi = 200)
    sys.stdout.close()