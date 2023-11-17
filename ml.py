#LINEAR REGRESSION
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
height=[[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]]
weight=[  16, 25 , 36, 49, 64, 81, 100]
plt.scatter(height,weight,color='black')
plt.xlabel("height")
plt.ylabel("weight")
reg=linear_model.LinearRegression()
reg.fit(height,weight)
X_height=[[12.0]]
print(reg.predict(X_height))


#NON LINEAR REGRESSION
import pandas as pd
x=[[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]]
y=[  16, 25 , 36, 49,64,81, 100]
# Step 2 - Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Step 4 Linear Regression prediction
print(lin_reg.predict([[11]]))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=1, include_bias=False),
    LinearRegression(),
)
polynomial_regression.fit(x,y)
X_height=[[20.0]]
target_predicted = polynomial_regression.predict(X_height)
print(target_predicted)
#DECISION TREE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
X = [[30],[40],[50],[60],[20],[10],[70]]
y = [0,1,1,1,0,0,1]
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X,y)
X_marks=[[20]]
print(classifier.predict(X_marks))

#NAIVE BAYES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:\\Misc\\Downloads\\Naive-Bayes-Classifier-Data.csv")
df.head()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
x=df.drop('diabetes',axis=1)
y=df['diabetes']
model=GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred


#LOGISTIC REGRESION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
X = [[30],[40],[50],[60],[20],[10],[70]]
y = [0,1,1,1,0,0,1]
classifier = LogisticRegression()
classifier.fit(X,y)
X_marks=[[20]]
print(classifier.predict(X_marks))
plt.show()