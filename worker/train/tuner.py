from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics  import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression



class Model_finder:
    def __init__(self):
        self.linearReg = LinearRegression()
        self.RandomForestReg = RandomForestRegressor()
        self.LogisticReg = LogisticRegression()
        self.RandomForestClsfr = RandomForestClassifier()

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):
        try:
            # Initializing with different combination of parameters
            self.param_grid_Random_forest_Tree = {
                                "n_estimators": [10,20,30],
                                "max_features": ["auto", "sqrt", "log2"],
                                "min_samples_split": [2,4,8],
                                "bootstrap": [True, False]
                                                     }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.RandomForestReg, self.param_grid_Random_forest_Tree, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.bootstrap = self.grid.best_params_['bootstrap']

            # creating a new model with the best parameters
            self.decisionTreeReg = RandomForestRegressor(n_estimators=self.n_estimators, max_features=self.max_features,
                                                         min_samples_split=self.min_samples_split, bootstrap=self.bootstrap)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            return self.decisionTreeReg
        except Exception as e:
            pass

    def get_best_params_for_linearReg(self, train_x, train_y):
        try:
            self.param_grid_linearReg = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]

            }

            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(self.linearReg,self.param_grid_linearReg, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.fit_intercept = self.grid.best_params_['fit_intercept']
            self.normalize = self.grid.best_params_['normalize']
            self.copy_X = self.grid.best_params_['copy_X']

            # creating a new model with the best parameters
            self.linReg = LinearRegression(fit_intercept=self.fit_intercept,normalize=self.normalize,copy_X=self.copy_X)
            # training the mew model
            self.linReg.fit(train_x, train_y)

            return self.linReg

        except Exception as e:
            pass

    def get_best_params_for_Random_Forest_Classifier(self, train_x, train_y):
        # try:
            print('sdjcbh')
            # Initializing with different combination of parameters
            self.param_grid_random_forest_classifier = {
                                "n_estimators": [8, 16, 32, 64],
                                "max_depth": [8, 16, 32, 64],
                                "max_features": [None, 'sqrt', 'log2']}

            # Creating an object of the Grid Search class
            self.grid = RandomizedSearchCV(self.RandomForestClsfr, self.param_grid_random_forest_classifier, verbose = 3, cv = 5, n_iter = 20)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']

            # creating a new model with the best parameters
            self.randomForestClsfr = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth,
                                                         max_features = self.max_features)
            # training the mew models
            self.randomForestClsfr.fit(train_x, train_y)
            return self.randomForestClsfr
        # except Exception as e:
        #     pass
        
    def get_best_params_for_Logistic_Regression(self, train_x, train_y):
        try:
            # Initializing with different combination of parameters
            self.param_grid_logistic = {
                                "penalty": ['l1', 'l2', 'elasticnet'],
                                "fit_intercept": [True, False],
                                "class_weight": [None, 'balanced']}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.LogisticReg, self.param_grid_logistic, verbose = 3, cv = 5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.penalty = self.grid.best_params_['penalty']
            self.fit_intercept = self.grid.best_params_['fit_intercept']
            self.class_weight = self.grid.best_params_['class_weight']

            # creating a new model with the best parameters
            self.logisticReg = LogisticRegression(penalty = self.penalty, fit_intercept = self.fit_intercept,
                                                         class_weight = self.class_weight)
            # training the mew models
            self.logisticReg.fit(train_x, train_y)
            return self.decisionTreeReg
        except Exception as e:
            pass
        
    def get_best_model(self, train_x, train_y, test_x, test_y):
        # try:
            if train_y.nunique() > 10:
                self.LinearReg = self.get_best_params_for_linearReg(train_x, train_y)
                self.prediction_LinearReg = self.LinearReg.predict(test_x) # Predictions using the LinearReg Model
                self.LinearReg_error = r2_score(test_y,self.prediction_LinearReg)


             # create best model for Randomforest Regressor
                self.randomForestReg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
                self.prediction_randomForestReg = self.randomForestReg.predict(test_x)  # Predictions using the randomForestReg Model
                self.prediction_randomForestReg_error = r2_score(test_y,self.prediction_randomForestReg)

                #comparing the two models
                if(self.LinearReg_error <  self.prediction_randomForestReg_error):
                    return 'RandomForestRegressor', self.randomForestReg
                else:
                    return 'LinearRegression', self.LinearReg
            else:
                self.RandomForestClassifier = self.get_best_params_for_Random_Forest_Classifier(train_x, train_y)
                self.prediction_RandomForestClassifier = self.RandomForestClassifier.predict(test_x) 
                self.prediction_randomForestClassifier_accuracy = accuracy_score(test_y, self.prediction_RandomForestClassifier)


             # create best model for Randomforest Regressor
                self.LogisticReg = self.get_best_params_for_Logistic_Regression(train_x, train_y)
                self.prediction_LogisticReg = self.LogisticReg.predict(test_x)  # Predictions using the randomForestReg Model
                self.prediction_LogisticReg_accuracy = accuracy_score(test_y, self.prediction_LogisticReg)

                #comparing the two models
                if(self.prediction_randomForestClassifier_accuracy <  self.prediction_LogisticReg_accuracy):
                    return 'LogisticRegression', self.LogisticReg
                else:
                    return 'RandomForestClassifier', self.RandomForestClassifier
        # except Exception as e:
        #     print('sdjcb')