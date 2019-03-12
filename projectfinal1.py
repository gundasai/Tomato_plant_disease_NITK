import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy
utils = importr("utils")
warble=importr("warbleR")

df = pd.read_csv('training_data.csv',header=0)
x = df[['meanfun' ,'minfun','maxfun']].values
y = df['label'].values
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.66,random_state=7)
model=SVC()
model.fit(x_train,y_train)
predictions=model.predict(x_validation)
print(accuracy_score(y_validation,predictions))
ro.r('dataframe <- data.frame(list = c("sound.files", "selec", "start", "end"))')

ro.r('dataframe <- data.frame("female.wav", 200, 1, 200)')

ro.r('names(dataframe) <- c("sound.files", "selec", "start", "end")')


ro.r('a <- specan(X=dataframe, bp=c(1,280),harmonicity = TRUE)')



values=ro.r('a')
x=numpy.array(values)
meanfun=(float)(x[36])
minfun=(float)(x[37])
maxfun=(float)(x[38])

print(model.predict([[meanfun,minfun,maxfun]]))
