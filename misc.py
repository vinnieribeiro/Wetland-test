import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem

def act(data):
    act = []
    for mol in data:
        act.append(mol.GetProp('NCI_AIDS_Antiviral_Screen_Conclusion'))
    return np.array(act,dtype=str)


def conf_table(pred,Act,classes):
    table = np.zeros((len(classes),len(classes)))
    
    for i,p in enumerate(pred):
#        print(p,Act[i])
        table[classes[p],classes[Act[i]]]+=1
    
    return table

def metrics(table):
    TP = table[1,1]
    TN = table[0,0]
    FP = table[1,0]
    FN = table[0,1]
    return (TP+TN)/(TP+TN+FP+FN),(TP)/(TP+FN),(TN)/(TN+FP),(TP)/(TP+FP),(TN)/(TN+FN)

def zscore(X,Y):
    N11 = sum(X[Y,:])
    N12 = sum(X)-N11
    n1 = sum(Y)
    n2 = len(Y)-n1
    p1 = N11/n1
    p2 = N12/n2
    p  = (N11+N12)/(n1+n2)
    z = (p1-p2)/np.sqrt(p*(1-p)*(1/n1+1/n2))
    return z
        

