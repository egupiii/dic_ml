import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



# Set command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, 
                                 description='classification pipeline')

parser.add_argument('--dataset')
parser.add_argument('--model',default="log_reg",type=str)



# Create a definition of a classification pipeline
def implement_classification_pipeline(args):
    # Get datasets
    df = pd.read_csv(args.dataset)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    # Split train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
    
    if args.model=="log_reg":
        y_pred, score = logistic_regression(X_train,X_test,y_train,y_test)
    elif args.model=="svm":
        y_pred, score = support_vector_machine(X_train,X_test,y_train,y_test)
    elif args.model=="dt":
        y_pred, score = decition_tree(X_train,X_test,y_train,y_test)
    else:
        print("Your model, {} did not work on this definition.".format(args.model))
    
    # Return the results
    print("y_pred is {}".format(y_pred))
    print("score is {}".format(score))
    


# Create a definition of logistic regression
def logistic_regression(X_train,X_test,y_train,y_test):
    # Initialize a classifier
    log_reg = LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial')
    # Fit the model according to the given training data
    log_reg.fit(X_train,y_train)
    # Predict class labels for samples in X_test
    y_pred = log_reg.predict(X_test)
    # Returns the mean accuracy on the given test data and labels
    score = log_reg.score(X_test,y_test)
    
    return y_pred,score



# Create a definition of SVM
def support_vector_machine(X_train,X_test,y_train,y_test):
    # Initialize a classifier
    svm = SVC(gamma='auto')
    # Fit the model according to the given training data
    svm.fit(X_train,y_train)
    # Predict class labels for samples in X_test
    y_pred = svm.predict(X_test)
    # Returns the mean accuracy on the given test data and labels
    score = svm.score(X_test,y_test)
    
    return y_pred,score



# Create a definition of decition tree
def decition_tree(X_train,X_test,y_train,y_test):
    # Initialize a classifier
    dt = DecisionTreeClassifier(random_state=0)
    # Fit the model according to the given training data
    dt.fit(X_train,y_train)
    # Predict class labels for samples in X_test
    y_pred = dt.predict(X_test)
    # Returns the mean accuracy on the given test data and labels
    score = dt.score(X_test,y_test)
    
    return y_pred,score



if __name__ == "__main__":
    # Be run here at first when running the py file

    # Import command-line arguments
    args = parser.parse_args()
    implement_classification_pipeline(args)