import numpy as np 

# function to create a confusion table
def conf_table(pred,Act,classes):
    table = np.zeros((len(classes),len(classes)))
    
    for i,p in enumerate(pred):
        table[classes[p],classes[Act[i]]]+=1
    
    return table

def metrics(table):
    TP = table[1,1]
    TN = table[0,0]
    FP = table[1,0]
    FN = table[0,1]
    return (TP+TN)/(TP+TN+FP+FN),(TP)/(TP+FN),(TN)/(TN+FP),(TP)/(TP+FP),(TN)/(TN+FN)



