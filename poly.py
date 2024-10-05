def poly_reg(df, degree=1):
    import pandas as pd
    import numpy as np
    acrs = pd.read_csv('data/acrs.csv') 
    X_test = acrs[['acr_size']]
    df  = df[df["Lot_Size"]<=80]

    df1 = df[(df["Price"]>900) & (df["Price"]<=1000000)]  
    #__________________________________________________________________________Removing Outilers by z score______________________________________________________________

    df1['zscore'] = (df1['Price'] - df1['Price'].mean())/df1['Price'].std()
    df1 = df1[(df1.zscore>-3) & (df1.zscore<3)]
    X = df1[['Lot_Size']]
    y_train = df1['Price']
     


   

    """### Feature Scaling"""

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X_train_ = sc.transform(X)
    X_test_ = sc.transform(X_test)

    """## Polynomial Linear Regression - ML Model Training"""

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree=degree)
    poly_reg.fit(X_train_)
    X_train_poly = poly_reg.transform(X_train_)
    X_test_poly = poly_reg.transform(X_test_)

    # X_train_poly.shape, X_test_poly.shape

    lr = LinearRegression()

    lr.fit(X_train_poly, y_train)

     

    # lr.predict([X_test_poly[0,:]])  #single value prediction.
    

    y_pred = lr.predict(X_test_poly)
    return y_pred  


def poly_reg_sqr(df, degree=1):
    import pandas as pd
    import numpy as np
    acrs = pd.read_csv('data/acrs.csv') 
    X_test = acrs[['acr_size']]*43560
    df  = df[df["Lot_Size"]<=80]

    df1 = df[(df["Price"]>900) & (df["Price"]<=1000000)]  
    #__________________________________________________________________________Removing Outilers by z score______________________________________________________________

    df1['zscore'] = (df1['Price'] - df1['Price'].mean())/df1['Price'].std()
    df1 = df1[(df1.zscore>-3) & (df1.zscore<3)]
    X = df1[['lot_sqft']]
    y_train = df1['Price']
     


   

    """### Feature Scaling"""

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X_train_ = sc.transform(X)
    X_test_ = sc.transform(X_test)

    """## Polynomial Linear Regression - ML Model Training"""

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree=degree)
    poly_reg.fit(X_train_)
    X_train_poly = poly_reg.transform(X_train_)
    X_test_poly = poly_reg.transform(X_test_)

    # X_train_poly.shape, X_test_poly.shape

    lr = LinearRegression()

    # lr.fit(X_train_poly, y_train)
    lr.fit(X, y_train)

     

    

    # y_pred = lr.predict(X_test_poly)
    y_pred = lr.predict(X_test)
    
    return y_pred  

#_________________________________________________SVR -> support vector regression________________________________________________________



def support_vector_reg(df, degree=2):
    import pandas as pd
    import numpy as np
    acrs = pd.read_csv('data/acrs.csv') 
    X_test = acrs[['acr_size']]
    df  = df[df["Lot_Size"]<=80]
    df = df[(df["Price"]>900) & (df["Price"]<=1000000)]  

    X = df[['Lot_Size']]
    y_train = df['Price']
    




    """### Feature Scaling"""

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X_train_ = sc.transform(X)
    X_test_ = sc.transform(X_test)

    """## Polynomial Linear Regression - ML Model Training"""
 
    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree=degree)
    poly_reg.fit(X_train_)
    X_train_poly = poly_reg.transform(X_train_)
    X_test_poly = poly_reg.transform(X_test_)

    X_train_poly.shape, X_test_poly.shape

    lr = LinearRegression()

    lr.fit(X_train_poly, y_train)

    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)
    # lr.predict([X_test_poly[0,:]])  #single value prediction.
    

    y_pred = regressor.predict(X_test_poly)
    return y_pred  