import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# Set command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, 
                                 description='linear regression pipeline')

parser.add_argument('--dataset')
parser.add_argument('--model',default="lr",type=str)



# Create a definition of a regression pipeline
def implement_regression_pipeline(args):
    # Get datasets
    df = pd.read_csv(args.dataset)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    # Split train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
    
    if args.model=="lr":
        y_pred, mse = linear_regression(X_train,X_test,y_train,y_test)
    else:
        print("Your model, {} did not work on this definition.".format(args.model))
    
    # Return the results
    print("y_pred is {}".format(y_pred))
    print("score is {}".format(score))



# Create a definition of linear regression
def linear_regression(X_train,X_test,y_train,y_test):
    # Initialize a classifier
    lr = LinearRegression()
    
    # Fit the model according to the given training data
    lr.fit(X_train,y_train)
    
    # Predict class labels for samples in X_test
    y_pred = lr.predict(X_test)
    
    # Returns the mean accuracy on the given test data and labels
    mse = mean_squared_error(y_test, y_pred)
    
    return y_pred, mse



if __name__ == "__main__":
    # Be run here at first when running the py file

    # Import command-line arguments
    args = parser.parse_args()
    implement_regression_pipeline(args)