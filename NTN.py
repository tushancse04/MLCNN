from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')
print ('Libraries Loaded')


class NeuralTensorLayer(Layer):
    
    def __init__(self, output_dim, input_dim, activation= None):
        super().__init__()
        self.output_dim = output_dim #The k in the formula
        self.input_dim = input_dim   #The d in the formula
        self.activation = activation #The f function in the formula
        
    def build(self, input_shape):
        #The initialisation parameters
        self.mean = 0.0 
        self.stddev = 1.0
        dtype = 'float32'
        self.seed = 1
        
        #The output and the inut dimension
        k = self.output_dim
        d = self.input_dim
        
        #Initialise the variables to be trained. The variables are according to the
        #function defined.
        self.W = K.variable(K.random_normal((k,d,d), self.mean, self.stddev,
                               dtype=dtype, seed=self.seed))
        self.V = K.variable(K.random_normal((2*d,k), self.mean, self.stddev,
                               dtype=dtype, seed=self.seed))
        self.b = K.zeros((self.input_dim,))
        
        #Set the variables to be trained.
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs):
        
        #Get Both the inputs
        e1 = inputs[0]
        e2 = inputs[1]
        
        #Get the batch size
        batch_size = K.shape(e1)[0]
        
        #The output and the inut dimension
        k = self.output_dim
        d = self.input_dim

        #The first term in the function which is the bilinear product is calculated here.
        first_term_k = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]
        for i in range(1, k):
            temp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
            first_term_k.append(temp)
        first_term = K.reshape(K.concatenate(first_term_k, axis=0), (batch_size, k))

        #The second term in the function is calculated here.
        second_term = K.dot(K.concatenate([e1,e2]), self.V)
        
        #Sum of the two terms to get the final function
        z =  first_term + second_term
        
        #The activation is selected here
        if (self.activation == None):
            return z
        elif (self.activation == 'tanh'):
            return K.tanh(z)
        elif (self.activation == 'relu'):
            return K.relu(z)
        else :
            print ('Activation not found')
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


    def fit(self,x_train1,x_train2,y_train,x_val1,x_val2,y_val):
        print(x_train1.shape,x_train2.shape,x_val1.shape,x_val2.shape,y_train.shape,y_val.shape)
        vector1 = Input(shape=(x_train1.shape[1],), dtype='float32')
        vector2 = Input(shape=(x_train1.shape[1],), dtype='float32')
        BilinearLayer = NeuralTensorLayer(output_dim=32, input_dim=x_train1.shape[1], 
                                          activation= 'tanh')([vector1, vector2])        


        #The g or the output of the modelled function.
        g = Dense(output_dim=1)(BilinearLayer)
        model = Model(input=[vector1, vector2], output=[g])

        #Compile the model
        adam = optimizers.adam(.001)
        model.compile( loss='mean_squared_error', optimizer=adam)
        #The summary of the model.
        #model.summary()
        model.fit([x_train1, x_train2], y_train, epochs=1,
                  validation_data=([x_val1, x_val2], y_val))

        r = model.predict([x_val1, x_val2])
        r = [x[0]  for x in r]
        s = roc_auc_score(y_val,r)
        r = [False if r[i] < .5 else True for i in range(len(r))]
        print('ntn : ',s)
        ofile = open('review/out/out.txt','a')
        ofile.write(str(s) + '\n')
        ofile.close()
        return r
        #print([y for y in y_val])
