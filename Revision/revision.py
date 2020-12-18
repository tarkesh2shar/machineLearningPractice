import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("India_population.csv")

#yfo=pd.read_csv("complete.csv")


X=dataset.iloc[:,0:1].values
Y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state=0)






#Our population dataset is nearly Ready lets add regression 
#and see if we need any scaler stuff or not

"""
    Simple Linear Regression//
"""
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train);

y_pred_linear_regression=regressor.predict(X_test)

plt.title("population vs Year(!! Linear Regression !!)")
plt.scatter(X_train,Y_train,color="red");
plt.xlabel("Year");
plt.ylabel("Population");
#regression 
plt.plot(X_train,regressor.predict(X_train),color="blue")



"""
Since there is just 1 variable here ,we dont need multiple linear Regression here 
Sorry !!!!

"""

from sklearn.preprocessing import PolynomialFeatures

featuurePoly=PolynomialFeatures(degree=5);

x_poly=featuurePoly.fit_transform(X_train);

regressorPoly=LinearRegression();

regressorPoly.fit(x_poly,Y_train);


y_pred_polynomial_regression=regressorPoly.predict(featuurePoly.fit_transform(X_test))


plt.title("population vs Year(!! Polynomial Regression !!)")
plt.scatter(X_train,Y_train,color="red");
plt.xlabel("Year");
plt.ylabel("Population");
#regression 
plt.plot(X_train,regressorPoly.predict( featuurePoly.fit_transform(X_train) ),color="blue");


#Allright then onto the next regression what is it ??

#SVR needs data to be scaled so scale it brother


from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

XScaled=scalerx.fit_transform(X_train)


scalery=StandardScaler();


scaledY=np.reshape(Y_train,(-1,1))

YScaled=scalery.fit_transform(scaledY);



from sklearn.svm import SVR;
regressorSVR=SVR(kernel="rbf");
regressorSVR.fit(XScaled,YScaled.ravel());



y_pred_svr = scalery.inverse_transform(
        
        regressorSVR.predict( 
                
                scalerx.transform([[1983]])
                
                )
        
        
        )

# Visualising the Regression results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(scalerx.transform(X_test), regressorSVR.predict(scalerx.transform(X_test)), color = 'blue')
plt.title('Population vs Year (SVRegression Model)')
plt.xlabel('Year')
plt.ylabel('Position')
plt.show()


#Some problem with the plot will fix it later for now lets go onto another
#regression model will ya ?

#here comes decision Tree Model hurray


from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)


regressor.fit(X_train,Y_train)

#non continous regresson model
y_pred_decision_tree=regressor.predict([[1983]])


X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Population vs Year (Decision Tree Regression Model)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

# hurray onto next 1 now  random Forest Here it Comes Now!!




from sklearn.ensemble import RandomForestRegressor



regressor=RandomForestRegressor(n_estimators=300,random_state=0)

regressor.fit(X_train,Y_train);




y_pred_random_tree=regressor.predict([[1983]])



X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Population vs Year (Random Forest Regression Model)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()











