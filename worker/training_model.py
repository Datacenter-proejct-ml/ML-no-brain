from sklearn.model_selection import train_test_split
import tuner
import pandas as pd
import pickle


class trainModel:
    def __init__(self):
        pass

    def trainingModel(self):
        try:
            # Splitting the data into training and test set for each cluster one by one
            df = pd.read_csv('cleaned.csv')
            x_train, x_test, y_train, y_test = train_test_split(df.drop('Output', axis = 1), df.Output, test_size=1 / 3, random_state=36)

            # model_finder = tuner.Model_finder.get_best_model(x_train, x_test, y_train, y_test)    #### Object intialization

            #### getting the best model 
            best_model_name, best_model = tuner.Model_finder().get_best_model(x_train, y_train, x_test, y_test)
            print(best_model)
            pickle.dump(best_model, open(f'{best_model}_model', 'wb'))
        except Exception:
            print(f'This is the error that we have received from the upper management {Exception}\nPlease try again later!!!')
        
a = trainModel().trainingModel()
