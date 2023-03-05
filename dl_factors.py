# -*- coding: utf-8 -*-



import os
import csv

import torch.cuda

os.chdir('/Users/hesterchan/Desktop/MFIN7034/HW3')

def load_csv(file_name):
    new_dict={}
    temp_stock_id=0
    with open(file_name,"r") as f:
        data=f.read()
        data1=data.split()
        #print(data1[0])
        #print(data1[-1])
        f.close()
    for i in range(1,len(data1)):
        temp=data1[i].split(",")
        if (temp_stock_id==0) or (temp_stock_id != temp[1]):
            if temp_stock_id ==0:
                pass
            else:
                new_dict[str(temp_stock_id)] = [temp_price, temp_return, temp_time]
            new_id=temp[1]
            temp_stock_id = new_id
            #clean the results
            #new_dict[str()] = []
            temp_price=[]
            temp_return=[]
            temp_time=[]
        #stockid+=[temp[1]]
        temp_price+=[float(temp[2])]
        temp_return+=[float(temp[3])]
        temp_time+=[float(temp[4])]
    return new_dict
        
    
#data2 contains all the necessary numbers here.
data2= load_csv("cleaned_data.csv")
#print(data2["10001"])

#load ff factors, create a iist of list

def load_ff(file_name):
    new_list=[]
    factor1=[]
    factor2=[]
    factor3=[]
    factor4=[]
    factor5=[]
    with open(file_name,"r") as f:
        data=f.read()
        #print("data")
        #print([data])
        data1=data.split("\n")
        #print(data1[0])
        #data2=data1.split(",")
        #print(data2)
    for i in range(len(data1)):
        line=data1[i]
        #print("line")
        #print(line)
        new_line = line.split(",")
        #print(new_line)
        #print("new line is")
        #print(new_line)
        factor1+=[float(new_line[1])]
        factor2+=[float(new_line[2])]
        factor3+=[float(new_line[3])]
        factor4+=[float(new_line[4])]
        factor5+=[float(new_line[5])]
    new_list = [factor1,factor2,factor3,factor4,factor5]
    return new_list

ff = load_ff("ff-factor.csv")
#print(len(ff[0]))
#print(ff[-1])

#now it is ready to perform dl

import torch as t

from torch.autograd import Variable as v 
#use the first 241 periods to train
training_ff =[] 
testing_ff=[]

for i in range(288):
    if i < 241:
        training_ff+= [[ff[0][i],ff[1][i],ff[2][i],ff[3][i],ff[4][i]]]
    else:
        testing_ff += [[ff[0][i],ff[1][i],ff[2][i],ff[3][i],ff[4][i]]]

#print(testing_ff)

#training_ff_tensor= v(t.FloatTensor(training_ff), requires_grad=True)
training_ff_tensor=t.FloatTensor(training_ff)
testing_ff_tensor=t.FloatTensor(testing_ff)

H=80


model = t.nn.Sequential( 
        t.nn.Linear(5, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        #t.nn.Dropout(p=0.4),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        #t.nn.Dropout(p=0.0),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        #t.nn.Dropout(p=0.3),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        #t.nn.Dropout(p=0.3),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        t.nn.Dropout(p=0.25),
        t.nn.ReLU(),
        #t.nn.Linear(H, H),
        #t.nn.BatchNorm1d(H),
        #t.nn.Dropout(p=0.3),
        #t.nn.ReLU(),
        t.nn.Linear(H, 1),
        ) 

loss_fn = t.nn.MSELoss()


#learning_rate=0.25
learning_rate=0.3
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
#ptimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs=5000
#one batch is enough, since the size of data is small
stock_ids=list(data2.keys())
#print(stock_ids)


import numpy as np
#just check one stock
R2=[]
R2_out=[]
Training_mse=[]
Testing_mse=[]
for i in range(0,241):
    current_stock = stock_ids[i]
    return_temp=data2[current_stock][1]
    #print(return_temp)
    return_train = t.FloatTensor(return_temp[0:241])
    return_test=t.FloatTensor(return_temp[241:288])
    for epoch in range(n_epochs):
        y_pred=model(training_ff_tensor)
        y_pred = t.reshape(y_pred,(-1,))
        #print(y_pred.shape)
        loss=loss_fn(y_pred,return_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    var_y = np.var(return_temp[0:241])
    R2.append(1-loss.item()/var_y)
    #out of sample
    y_pred1=t.reshape(model(testing_ff_tensor),(-1,))
    loss1=loss_fn(y_pred1, return_test)
    var_y1= np.var(return_temp[241:288])
    R2_out.append( 1-loss1.item()/var_y1)
    Training_mse.append(loss.item())
    Testing_mse.append(loss1.item())

print("The avg in sample R2:", np.mean(np.array(R2)))
print("The avg in sample MSE:", np.mean(np.array(Training_mse)))
print("The avg out sample R2:",np.mean(np.array(R2_out)))
print("The avg out sample MSE:",np.mean(np.array(Testing_mse)))



