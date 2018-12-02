## This code reads an array of land cover pixels (raster). Rasters of 100 x 100 pixels. 
## Each pixel is about 30 m length. The raster (.csv file) that defines what is a wetland is called 'wetlands' and it contains 5 types:
#nan: no wetland
#1: Lake
#2:Emergent
#3:Forested wetlands
#4:Ponds
#5:Peatland 

## The code predicts the presence of a peatland using reflection of different bands of Landsat images.

# Importing basic libraries
import pandas as pd
import numpy as np

#Importing models and preprocessing tools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,KFold
from misc import conf_table
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Importing balancing tools
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,ClusterCentroids
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.combine import SMOTETomek


# Reading CSV files
wetlands = np.array(pd.read_csv('wetlands.csv')).reshape(-1)

bands = []
bands.append(np.array(pd.read_csv('aerosol.csv'),dtype=float).reshape(-1))
bands.append(np.array(pd.read_csv('flags.csv'),dtype=float).reshape(-1))

for i in range(1,10):
    bands.append(np.array(pd.read_csv('Band'+str(i)+'.csv'),dtype=float).reshape(-1))


#Formatting the data into Numpy array:
bands = np.array(bands).transpose()
scaler = StandardScaler()

# converting all nan values to 0
wetlands[np.isnan(wetlands)] = 0
rlv = wetlands!=0 #Creates a mask to ignore the no wetland label

#X = scaler.fit_transform(bands)[rlv] #Standardizing the bands also ignores no wetland label
X = scaler.fit_transform(bands) #Standardizing the bands

# Class labels:
#0: no wetland 
#1: Lakes
#2:Emergent
#3:Forested wetlands
#4:Ponds
#5:Peatland

#Original labeling
#types = {0:'N', 1: 'L',2:'E',3:'F',4:'p',5:'P'}

# This labeling schemes merges Lakes and ponds
types = {0:'N', 1: 'L',2:'E',3:'F',4:'L',5:'P'}


# Organizing the response vector:
Y = np.empty(len(X),dtype =str)

#for i,y in enumerate(wetlands[rlv]): #ignores no wetland label
for i,y in enumerate(wetlands):
    Y[i] = types[y]
   

# Initial Modeling opitions:
# You can comment and uncomment to choose the modeling method being used
#model = LogisticRegression()
#model =  MLPClassifier()
#model = KNeighborsClassifier() #KNN is a simple model and seems to work well.
model = SVC(gamma='auto')# seems to work better with over sampling
#model = RandomForestClassifier()
#model = LinearDiscriminantAnalysis()

kf = StratifiedKFold(n_splits = 5, shuffle=True) #Generating the training test splits for a 5-fold crossvalidation

# Performing 5-fold cross-validation

#table = np.zeros((5,5)) #Creating a matrix o store the confusion table 
table = np.zeros((6,6))

for train,test in kf.split(X,Y):
    X_train = X[train]
    X_test = X[test]
    
    Y_train = Y[train]
    Y_test = Y[test]

# Choosing resampling method    
#    resampler = RandomUnderSampler()
#    resampler = RandomOverSampler()
#    resampler = TomekLinks()
#    resampler = ClusterCentroids()# for some reason clustercentroids+ RF seems to work well for peatlands
    resampler = SMOTE()
#    resampler = SMOTETomek()

    X_sample, Y_sample = resampler.fit_sample(X_train, Y_train)
    
    model.fit(X_sample,Y_sample)
        
#    table+=conf_table(model.predict(X_test),Y_test,{'L':0,'E':1,'F':2,'p':3,'P':4}) #This function generates a confusion matrix ignoring no wetland labels
#    table+=conf_table(model.predict(X_test),Y_test,{'N':0, 'L':1,'E':2,'F':3,'p':4,'P':5}) #This function generates a confusion matrix using all labels
    table+=conf_table(model.predict(X_test),Y_test,{'N':0, 'L':1,'p':1,'E':2,'F':3,'P':4}) #This function generates a confusion matrix merging lakes and ponds
    

# printing precision for each category.
#print(table[0,0]/sum(Y=='L'),table[1,1]/sum(Y=='E'),table[2,2]/sum(Y=='F'),table[3,3]/sum(Y=='p'),table[4,4]/sum(Y=='P')) 
#print(table[0,0]/sum(Y=='N'),table[1,1]/sum(Y=='L'),table[2,2]/sum(Y=='E'),table[3,3]/sum(Y=='F'),table[4,4]/sum(Y=='p'),table[5,5]/sum(Y=='P'))
print(table[0,0]/sum(Y=='N'),table[1,1]/(sum(Y=='L')+sum(Y=='p')),table[2,2]/sum(Y=='E'),table[3,3]/sum(Y=='F'),table[4,4]/sum(Y=='P'))