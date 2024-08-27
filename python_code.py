import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
#* Reading main direcotry and creating dataframes for train and test data
directory = "./X-RAY"
Train_data = pd.DataFrame(columns=['path', 'class'])
Test_data = pd.DataFrame(columns=['path', 'class'])


for filename in os.listdir(directory):
    for filename2 in os.listdir(directory+'/'+filename):
        for images in os.listdir(directory+'/'+filename+'/'+filename2):
            if filename =='train':
                Train_data = Train_data.append({'path': directory+'/'+filename+'/'+filename2+'/'+images , 'class': filename2}, ignore_index=True)
            else :
                Test_data = Test_data.append({'path': directory+'/'+filename+'/'+filename2+'/'+images , 'class': filename2}, ignore_index=True)

        Train_data
np.unique(Train_data['class'])
Test_data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
for i in range(462, 466):
    title = Test_data['class'][i]
    path = './X-RAY/test/COVID19/COVID19({}).jpg'
    image = mpimg.imread(path.format(i))
    plt.title('COVID19')
    plt.show()
    plt.imshow(image)

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GRU
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization 
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#* Defining parameters
batch_size = 24
size = (224,224,3)
img_width = img_hight = size[0]
clases = ['COVID19', 'NORMAL', 'PNEUMONIA']

#* Creating data generators
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#* Creating train and validation data generators
#* path -> from the dataframe
#* x_col -> the column name that contains the path,
#* y_col -> the column name that contains the class
#* image_size -> the size of the image
#* target_size -> the size of the image that will be used for training
#* color_mode -> the color mode of the image
#* batch_size -> the size of the batch
#* class_mode -> the type of the class
#* classes -> the classes that will be used
#* subset -> the subset of the data that will be used
#* shuffle -> if the data will be shuffled or not

train_data = data_gen.flow_from_dataframe(Train_data, x_col='path', y_col='class',
                                              image_size=(img_hight, img_width), target_size=(
                                                  img_hight, img_hight), color_mode='rgb',
                                              batch_size=batch_size, class_mode='categorical',
                                              classes=clases, subset='training')

val_data = data_gen.flow_from_dataframe(Train_data, x_col='path', y_col='class',
                                              image_size=(img_hight, img_width), target_size=(
                                                  img_hight, img_hight), color_mode='rgb',
                                              batch_size=batch_size, class_mode='categorical',
                                              classes=clases, subset='validation')
test_data = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                              image_size=(img_hight, img_width), target_size=(
                                                  img_hight, img_hight), color_mode='rgb',
                                              batch_size=batch_size, class_mode='categorical',
                                              classes=clases, subset=None)

###################################UnSHuffled#########################################
#* will be used for the confusion matrix analysis for results 
 
test_data_unshuffled = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                              image_size=(img_hight, img_width), target_size=(
                                                  img_hight, img_hight), color_mode='rgb',
                                              batch_size=batch_size, class_mode='categorical',
                                              classes=clases, subset=None, shuffle=False)



def Create_model(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):

    # * Create the model
    model = keras.Sequential()

    # * configure the inputshape
    model.add(keras.Input(shape=Image_shape))

    # * Add the first block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block2, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block2, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the third block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the fourth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the fifth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add((MaxPooling2D(pool_size=(2, 2))))

    # * Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((7*7, 512)))
    model.add(GRU(512, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    #* flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())
    
    # * Output layer
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model

#* compile function
def model_compiling(model, loss = 'categorical_crossentropy', optimizer = 'adam'):
    model.compile(
        #loss =keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

def Create_model_2(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):

    # * Create the model
    model = keras.Sequential()

    # * configure the inputshape
    model.add(keras.Input(shape=Image_shape))

    # * Add the first block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the second block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the third block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block4, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the fourth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block5, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add((MaxPooling2D(pool_size=(2, 2))))

    # * Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((7*7, 512)))
    model.add(GRU(512, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    #* flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())
    
    # * Output layer
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model


def Create_model_4(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):

    # * Create the model
    model = keras.Sequential()

    # * configure the inputshape
    model.add(keras.Input(shape=Image_shape))

    # * Add the first block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # * Add the second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Adding additional blocks
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(1024, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(1024, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

#     # * Add the third block
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     # * Add the fourth block
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
# #     model.add((MaxPooling2D(pool_size=(2, 2))))

    # * Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((7*7, 1024)))
    model.add(GRU(512, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    #* flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())
    
    # * Output layer
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model

def Create_model_3(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):

    # * Create the model
    model = keras.Sequential()

    # * configure the inputshape
    model.add(keras.Input(shape=Image_shape))

    # * Add the first block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # * Add the second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block2, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
              trainable=block2, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # * Add the third block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
              trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

#     # * Add the fourth block
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block4, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.3))

#     # * Add the fifth block
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#               trainable=block5, kernel_regularizer=regularizer))
#     model.add(BatchNormalization())
#     model.add((MaxPooling2D(pool_size=(2, 2))))
#     # model.add(Dropout(0.3))

    # * Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((28*28, 256)))
    model.add(GRU(256, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    #* flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())
    
    # * Output layer
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model

model_path = "Model_2_Acc_based.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_path,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 15,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=6, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)
model_path_2 = "Model_2_Acc_based_2.h5"
checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_path_2,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 15,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=6, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)
model_path_3 = "Model_2_Acc_based_3.h5"
checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_path_3,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 15,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=6, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)
model_path_4 = "Model_2_Acc_based_4.h5"
checkpoint_4 = tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_path_4,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 15,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=6, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)

model1 = Create_model(size)
model_compiling(model1, loss = 'categorical_crossentropy', optimizer = 'adam')
model2 = Create_model_2(size)
model_compiling(model2, loss = 'categorical_crossentropy', optimizer = 'adam')
model3 = Create_model_3(size)
model_compiling(model3, loss = 'categorical_crossentropy', optimizer = 'adam')
model4 = Create_model_4(size)
model_compiling(model4, loss = 'categorical_crossentropy', optimizer = 'adam')


# history1 = model1.fit(
#     train_data, 
#     validation_data= val_data, 
#     epochs=4, 
#     callbacks=[earlystop, checkpoint, learning_rate_reduction])
# history2 = model2.fit(
#     train_data, 
#     validation_data= val_data, 
#     epochs=10, 
#     callbacks=[earlystop, checkpoint_2, learning_rate_reduction])
# history3 = model3.fit(
#     train_data, 
#     validation_data= val_data, 
#     epochs=10, 
#     callbacks=[earlystop, checkpoint_3, learning_rate_reduction])
# history4 = model4.fit(
#     train_data, 
#     validation_data= val_data, 
#     epochs=10, 
#     callbacks=[earlystop, checkpoint_4, learning_rate_reduction])

# pd.DataFrame(history1.history)
# pd.DataFrame.to_csv(pd.DataFrame(history1.history), 'history1_acc_monitoring.csv', index=False)
# pd.DataFrame(history2.history)
# pd.DataFrame.to_csv(pd.DataFrame(history2.history), 'history2_acc_monitoring.csv', index=False)
# pd.DataFrame(history3.history)
# pd.DataFrame.to_csv(pd.DataFrame(history3.history), 'history3_acc_monitoring.csv', index=False)
# pd.DataFrame(history4.history)
# pd.DataFrame.to_csv(pd.DataFrame(history4.history), 'history4_acc_monitoring.csv', index=False)


# !pip install gdown
# import gdown

# #* Download the model weights and history
# url = "https://drive.google.com/file/d/1MOBkvpaCEg6bJ_3MPKO7j8wFOHCyrNHl/view?usp=share_link"
# output = "model.h5"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1B10MoA1PTb8FtCQzTxlKFwhK4iBAtTT7/view?usp=share_link"
# output = "Model_history.csv"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

model1 = Create_model(size)
model_compiling(model1, loss = 'categorical_crossentropy', optimizer = 'adam')
model2 = Create_model_2(size)
model_compiling(model2, loss = 'categorical_crossentropy', optimizer = 'adam')
model3 = Create_model_3(size)
model_compiling(model3, loss = 'categorical_crossentropy', optimizer = 'adam')
model4 = Create_model_4(size)
model_compiling(model4, loss = 'categorical_crossentropy', optimizer = 'adam')

model1.load_weights('./Model_2_Acc_based.h5')
model2.load_weights('./Model_2_Acc_based_2.h5')
model3.load_weights('./Model_2_Acc_based_3.h5')
model4.load_weights('./Model_2_Acc_based_4.h5')

pred1 = model1.predict(test_data)
pred2 = model2.predict(test_data)
pred3 = model3.predict(test_data)
pred4 = model4.predict(test_data)

model1.evaluate(test_data)

# model.evaluate(test_data_unshuffled)

# predictions of the model on the unshuffled test set
# predictions = model.predict(test_data_unshuffled, verbose=1 ,
#                             workers=5, use_multiprocessing=True)
# predictions.shape


class_dict_1 = test_data.class_indices
class_dict_1 = {value: key for key, value in class_dict_1.items()} 
predicted_classes_1 = [class_dict_1.get(list(pred1[i]).index(pred1[i].max())) for i in range(len(pred1))]
Test_data['predicted_class_1'] = predicted_classes_1
Test_data.sample(15)

class_dict_2 = test_data.class_indices
class_dict_2 = {value: key for key, value in class_dict_2.items()} 
predicted_classes_2 = [class_dict_2.get(list(pred2[i]).index(pred2[i].max())) for i in range(len(pred2))]
Test_data['predicted_class_2'] = predicted_classes_2
Test_data.sample(15)

class_dict_3 = test_data.class_indices
class_dict_3 = {value: key for key, value in class_dict_3.items()} 
predicted_classes_3 = [class_dict_3.get(list(pred3[i]).index(pred3[i].max())) for i in range(len(pred3))]
Test_data['predicted_class_3'] = predicted_classes_3
Test_data.sample(15)

class_dict_4 = test_data.class_indices
class_dict_4 = {value: key for key, value in class_dict_4.items()} 
predicted_classes_4 = [class_dict_4.get(list(pred4[i]).index(pred4[i].max())) for i in range(len(pred4))]
Test_data['predicted_class_4'] = predicted_classes_4
Test_data.sample(15)
Test_data

S_Data = pd.DataFrame()
S_Data['pm_1'] = Test_data['predicted_class_1']
S_Data['pm_2'] = Test_data['predicted_class_2']
S_Data['pm_3'] = Test_data['predicted_class_3']
S_Data['pm_4'] = Test_data['predicted_class_4']
S_Data['target'] = Test_data['class']

np.unique(S_Data['target'])

S_Data = S_Data.replace('COVID19', 0)
S_Data = S_Data.replace('NORMAL', 1)
S_Data = S_Data.replace('PNEUMONIA', 2)
display(S_Data)

from sklearn.metrics import f1_score
print(f1_score(S_Data['target'], S_Data['pm_1'], average=None))
print(f1_score(S_Data['target'], S_Data['pm_2'], average=None) )
print(f1_score(S_Data['target'], S_Data['pm_3'], average=None))
print(f1_score(S_Data['target'], S_Data['pm_4'], average=None))

!pip install lazypredict

import lazypredict
from sklearn.model_selection import train_test_split

X = S_Data.iloc[:, :-1]
y = S_Data.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.1,random_state =123)

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

# print("Correct prediction: " ,len(Test_data[Test_data['matched' ]== True]))
# print("False prediction: " ,len(Test_data[Test_data['matched' ]== False]))
# print("Accucary of the model on unseen data " , round(len(Test_data[Test_data['matched' ]== True])/len(Test_data),4), "%")

from sklearn.metrics import confusion_matrix
import plotly.express as px
conf_mat = confusion_matrix(Test_data['class'], 
                            Test_data['predicted_class'],
                            labels = list(class_dict.values()))

def confusion_matrix_plot(conf_mat,labels = list(class_dict.values()),length = len(Test_data)):
    fig = px.imshow(conf_mat,
                    labels=dict(x="Actual-class", y="Predicted-class"), color_continuous_scale='viridis',
                    x=labels,
                    y=labels,
                    text_auto=True, aspect="auto", range_color = [0,length]
                   )
    fig.update_xaxes(side="top")
    fig.update_layout(font=dict(color='black'))
    fig.show()
confusion_matrix_plot(conf_mat,labels = list(class_dict.values()),length = len(Test_data))
! when saving plot version it doesn't show any text--> kaggle backend problem


#mport seaborn as sns
#ns.heatmap(conf_mat, annot=True, linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True), fmt='.0f')

