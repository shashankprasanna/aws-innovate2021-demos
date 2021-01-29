import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization

NUM_CLASSES = 10

class CustomModel(Model):
    def __init__(self, input_shape):
        super(CustomModel, self).__init__()
        self.conv1a = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)
        self.bn1a = BatchNormalization()
        self.act1a = Activation('relu')
        self.conv1b = Conv2D(32, (3, 3))
        self.bn1b = BatchNormalization()
        self.act1b = Activation('relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.2)
               
        self.conv2a = Conv2D(64, (3, 3), padding='same', input_shape=input_shape)
        self.bn2a = BatchNormalization()
        self.act2a = Activation('relu')
        self.conv2b = Conv2D(64, (3, 3))
        self.bn2b = BatchNormalization()
        self.act2b = Activation('relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = Dropout(0.3)
        
        self.conv3a = Conv2D(128, (3, 3), padding='same', input_shape=input_shape)
        self.bn3a = BatchNormalization()
        self.act3a = Activation('relu')
        self.conv3b = Conv2D(128, (3, 3))
        self.bn3b = BatchNormalization()
        self.act3b = Activation('relu')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = Dropout(0.4)
        
        self.flatten4 = Flatten()
        self.dense4a = Dense(256)
        self.act4a = Activation('relu')
        self.dropout4 = Dropout(0.5)
        self.dense4b = Dense(NUM_CLASSES)
        self.act4b = Activation('softmax')
        
    def call(self, x):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.act1a(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.act1b(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.act2a(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.act2b(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.act3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.act3b(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.flatten4(x)
        x = self.dense4a(x)
        x = self.act4a(x)
        x = self.dropout4(x)
        x = self.dense4b(x)
        x = self.act4b(x)
        
        return x
        
def get_custom_model(input_shape):
    model = CustomModel(input_shape)
    return model

