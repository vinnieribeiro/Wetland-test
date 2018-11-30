import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,KFold
from misc import conf_table
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Reading CSV files
wetlands = np.array(pd.read_csv('wetlands.csv')).reshape(-1)

bands = []
bands.append(np.array(pd.read_csv('aerosol.csv')).reshape(-1))
bands.append(np.array(pd.read_csv('flags.csv')).reshape(-1))

for i in range(1,10):
    bands.append(np.array(pd.read_csv('Band'+str(i)+'.csv')).reshape(-1))


#Formatting the data into Numpy array:
bands = np.array(bands).transpose()
scaler = StandardScaler()

wetlands[np.isnan(wetlands)] = 0
rlv = wetlands!=0

X = scaler.fit_transform(bands)[rlv] #Standardizing the bands
#X = scaler.fit_transform(bands) #Standardizing the bands


# Class labels:
#nan: no wetland
#1: Lake
#2:Emergent
#3:Forested wetlands
#4:Ponds
#5:Peatland

types = {0:'N', 1: 'L',2:'E',3:'F',4:'p',5:'P'}


# Organizing the response vector:
Y = np.empty(len(X),dtype =str)

for i,y in enumerate(wetlands[rlv]):
#for i,y in enumerate(wetlands)[rlv]):
    Y[i] = types[y]
   

# Initial Modeling opitions:
# You can comment and uncomment to choose the modeling method being used
#model = LogisticRegression()
#model =  MLPClassifier()
model = KNeighborsClassifier() #KNN is a simple model and seems to work well.
#model = SVC()
#model = RandomForestClassifier()
#model = LinearDiscriminantAnalysis()

kf = StratifiedKFold(n_splits = 5, shuffle=True) #Generating the training test splits for a 5-fold crossvalidation

table = np.zeros((5,5))
#table = np.zeros((6,6))
for train,test in kf.split(X,Y):
    X_train = X[train]
    X_test = X[test]
    
    Y_train = Y[train]
    Y_test = Y[test]
    
    
    model.fit(X_train,Y_train)
        
    table+=conf_table(model.predict(X_test),Y_test,{'L':0,'E':1,'F':2,'p':3,'P':4}) #This function generates a confusion matrix
#    table+=conf_table(model.predict(X_test),Y_test,{'N':0, 'L':1,'E':2,'F':3,'p':4,'P':5}) #This function generates a confusion matrix


# printing the results according to the 
print(table[0,0]/sum(Y=='L'),table[1,1]/sum(Y=='E'),table[2,2]/sum(Y=='F'),table[3,3]/sum(Y=='p'),table[4,4]/sum(Y=='P'))
#print(table[0,0]/sum(Y=='N'),table[1,1]/sum(Y=='L'),table[2,2]/sum(Y=='E'),table[3,3]/sum(Y=='F'),table[4,4]/sum(Y=='p'),table[5,5]/sum(Y=='P'))
