
# coding: utf-8

# In[2]:


import xlrd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math


#----------loss function:RMSE----------------
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def train_model(X_data,Y_data,round):
    
    #----------model---------------
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    model = Sequential()
    model.add(Dense(units = 1000,input_dim = 30000,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 750,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 500,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 350,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 250,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 200,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 150,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 25,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 1,kernel_initializer='uniform',activation='linear'))
    
    model.compile(loss=root_mean_squared_error,
                 optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=X_data,y=Y_data,validation_split = 0.2,epochs=100,batch_size=10,verbose=2)
    
    #---------test validation------------
    predict = model.predict(X_data)
    loss = 0
    for i in range(8):
        print(predict[32+i] , Y_data[32+i])
        loss += math.pow(predict[32+i] - Y_data[32+i],2)
    loss = math.sqrt(loss/8)
    print(loss)
    if loss <= 0.15:
        model.save('f_weights//my_model_'+str(round)+'.h5') 
        return 1
    else:
        return 0

def load_model(my_model):
    from keras.models import load_model
    model = load_model('f_weights//'+str(my_model),custom_objects={'root_mean_squared_error': root_mean_squared_error})
    return model



    
if __name__== "__main__":
    
    dataset = [None]*40
    path = "C://Users//TsungYuan//Desktop//BDA//806初賽訓練數據"
    files= os.listdir(path)
    i = 0
    for file in files: 
         if not os.path.isdir(file):
             dataset[i] = file
             i += 1
    #print(dataset)
    
    X_data = np.zeros( (40,30000) )
    Y_data = np.zeros( (40,1) )
    for i in range(40):
        x_data = np.zeros( (7500,4) )
        y_data = 0
    
    #---------讀資料夾所有檔案--------
        b =xlrd.open_workbook("806初賽訓練數據//"+dataset[i])
        s = b.sheet_by_index(0)
    
        for line in range(7500):
            x_data[line] = s.row_values(line,0,4)
        #print(x_data)
    
    #---------做normalization---------
        scaler = MinMaxScaler()
        scaler.fit(x_data)
        x_data = scaler.transform(x_data)
        #print(x_data)
        
        x_data = x_data.reshape(1,-1)
        X_data[i] = x_data
        #print(x_data)
    
    #----------(40,30000)-------------    
        y = str(s.row_values(7500,0,1))
        y = y[2:-2]
        y_data = y.strip().split(":")[1]
        Y_data[i] = y_data
        #print(y_data)
        
        
    #----------train_model-------------
    for round in range(60):
        random = np.arange(X_data.shape[0])
        np.random.shuffle(random)
        X_data = X_data[random]
        Y_data = Y_data[random]
        
        good_model = train_model(X_data,Y_data,round) 
        print("training finish")
        
 #------------------testdata-----------   
    testset = [None]*11
    Testset = [None]*10
    testpath = "C://Users//TsungYuan//Desktop//BDA//831測驗集"
    files= os.listdir(testpath)
    i = 0
    for file in files: 
       if not os.path.isdir(file):
           testset[i] = file
           i += 1
    temp = testset[0]
    testset[10] = temp
    Testset[:] = testset[1:11]
    #print(Testset)
        
    X_test = np.zeros( (10,30000) )
    
    for i in range(10):
       x_test = np.zeros( (7500,4) )
       y_test = 0
    #---------讀資料夾所有檔案--------
       b =xlrd.open_workbook("831測驗集//"+Testset[i])
       s = b.sheet_by_index(0)
       for line in range(7500):
           x_test[line] = s.row_values(line,0,4)
           #print(x_test)
       #print(x_test)
        #---------做normalization---------
       scaler = MinMaxScaler()
       scaler.fit(x_test)
       x_data = scaler.transform(x_test)
       #print(x_data)
            
       x_test = x_test.reshape(1,-1)
       X_test[i] = x_test
    #print(X_test)
    
    #---------------load_model---------
    count = 0  
    modelset = [None]*100
    path = "C://Users//TsungYuan//Desktop//BDA//f_weights"
    files= os.listdir(path)
    i = 0
    for file in files: 
         if not os.path.isdir(file):
             modelset[i] = file
             i += 1
             count += 1
    #print(modelset,count)
    for j in range(count):
        model = load_model(modelset[j])
        predict = model.predict(X_test)
        for i in range(10):
            print(Testset[i],predict[i])
        predict = model.predict(X_data)
        loss = 0
        for i in range(8):
            print(predict[32+i] , Y_data[32+i])
            loss += math.pow(predict[32+i] - Y_data[32+i],2)
        loss = math.sqrt(loss/8)
        print(modelset[j]," : ",loss)
