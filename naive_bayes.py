from locale import normalize

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import metrics
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from reader import Reader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import normaltest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

pd.options.mode.chained_assignment = None  # default='warn'


class main(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data

    def percentile(self):
        x = self.df['SalePrice']
        threshold = x.quantile([0.33,0.67])
        self.firstRange, self.secondRange = threshold.iloc[0], threshold.iloc[1]

        return self.firstRange, self.secondRange

    def data_classification(self):
        df = self.df

        column_names = ['SalePrice','LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath']
        df = df[column_names]
       
        df.dropna(subset=column_names, inplace=True)

        return df
        
    def groupBy_ResponseVar(self):

        fR, sR = self.percentile()
        df = self.data_classification()

        df['SaleRange'] = df['SalePrice'].apply(
            lambda x: 'Low' if x <= fR 
            else ('Medium' if (x > fR and x <= sR) else 'High'))
        df_balance = df.copy()

        # df = df.groupby('SaleRange').size()

        df_balance['SaleRange'] = df_balance['SaleRange'].astype('category')
       
        return df

    def train_test(self):
        df = self.groupBy_ResponseVar()
        y = df.pop('SaleRange')
        X = df[['LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath']]

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)

        return  X_train, X_test, y_train, y_test, X, y


    def normalizeData(self):
        X_train, X_test,y_train, y_test, X, y = self.train_test()
        
        model = make_pipeline(StandardScaler(), LogisticRegression())
        cv_result = cross_validate(model,X_train, y_train, cv=5 )
        return cv_result


    def treeDepth(self):
        
        X_train, X_test,y_train, y_test, X, y = self.train_test()

        train_accuracy = []
        test_accuracy = []

        for depth in range(1, 10):
            df = tree.DecisionTreeClassifier(max_depth=depth, random_state=10)
            df.fit(X_train, y_train)
            
            train_accuracy.append(df.score(X_train, y_train))
            test_accuracy.append(df.score(X_test, y_test))

            

        frame = pd.DataFrame({'max_depth':range(1, 10), 'train_acc':train_accuracy, 'test_acc':test_accuracy})
        print(frame.head())


        # EL DEPTH ES DE 3

    def decision_tree(self):
        # CAMBIOOOOO2
        X_train, X_test,y_train, y_test, X, y = self.train_test()
       

        dt = tree.DecisionTreeClassifier(max_depth=3, random_state=10)
        dt.fit(X_train, y_train)

        column_names = ['LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath']
        tree.export_graphviz(dt, out_file='tree.dot', feature_names=column_names, class_names=True, max_depth=2)
        

        y_pred = dt.predict(X_test)
        print ("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print ("Precision:", metrics.precision_score(y_test,y_pred,average="weighted", zero_division=1) )
        print ("Recall: ", metrics.recall_score(y_test,y_pred,average="weighted", zero_division=1))


        # para correrlo tiene que descargar graphviz
        # despues -> dot -Tpng tree.dot -o tree.png
    
    def regression_tree(self):
        
        X_train, X_test,y_train, y_test, X, y = self.train_test()
       

        rt = tree.DecisionTreeRegressor(max_depth=3, random_state=10)
        rt.fit(X_train, y_train)

        column_names = ['LotArea','OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'FullBath']
        tree.export_graphviz(rt, out_file='regression_tree.dot', feature_names=column_names, class_names=True, max_depth=2)

    def random_forest(self):
        #CAMBIOOO2
        X_train, X_test,y_train, y_test, X, y = self.train_test()
       
        rf = RandomForestClassifier(max_depth=3, random_state=10)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        print ("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print ("Precision:", metrics.precision_score(y_test,y_pred,average="weighted", zero_division=1) )
        print ("Recall: ", metrics.recall_score(y_test,y_pred,average="weighted", zero_division=1))
        

    # CAMBIOOOOOOO
    def linear_regression(self):
        X_train, X_test,y_train, y_test, X, y = self.train_test()
        y_tr = y_train.values.reshape(-1, 1)
        y_t = y_test.values.reshape(-1, 1)

        x_tr = X_train['OverallQual'].values.reshape(-1, 1)
        x_t = X_test['OverallQual'].values.reshape(-1, 1)

        lm = LinearRegression()
        #  cambioo
        lm.fit(x_tr, y_tr)
        y_pred = lm.predict(x_tr)
        
        m = lm.coef_[0][0]
        c = lm.intercept_[0]

        

        print("Mean Squared Error: %.2f"%mean_squared_error(y_tr, y_pred))
        print("R2: %.2f"%r2_score(y_tr, y_pred))

        est = sm.OLS(y_tr,x_tr)
        est2 = est.fit()
        print(est2.summary())

        plt.scatter(y_tr, x_tr,  s=2)
        plt.plot(y_pred, x_tr, color="blue", markersize=2)
        plt.xlabel('OverallQual')
        plt.ylabel('SalePrice')
        plt.title("Conjunto de entrenamiento OverallQual vs SalePrice")
        plt.show()

    def multicollinearity(self):
        X_train, X_test,y_train, y_test, X, y = self.train_test()
        
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X, i) for i in range(len(X.columns))]

        print(vif_data)
    def residualAndSize(self, y_t, y_pred):
        residuales = y_t - y_pred
        return len(residuales), residuales
    
    def residualPlot(self, x_t, residuales):
        plt.plot(x_t,residuales, 'o', color='darkblue')
        plt.title("Gráfico de Residuales")
        plt.xlabel("Variable independiente")
        plt.ylabel("Residuales")

    def residualDist(self, residuales):
        sns.distplot(residuales)
        plt.title("Residuales")

    def residualBox(self, residuales):
        plt.boxplot(residuales)

    def residualNormal(self, residuales):
        return normaltest(residuales)
    
    def qualityModel(self, y_t, y_pred, x_t):
        residualLen, residuales = self.residualAndSize(self, y_t, y_pred)
        self.residualPlot(x_t, residuales)
        #self.residualDist(residuales)
        #self.residualBox(residuales)
        #self.residualNormal(residuales)

    def naive_bayes(self):
        X_train, X_test,y_train, y_test, X, y = self.train_test()

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred  =  classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        accuracy=accuracy_score(y_test,y_pred)

        print('Confusion matrix for Naive Bayes\n',cm)
        print('Accuracy: ',accuracy)


driver = main('train.csv')


print(driver.naive_bayes())
    