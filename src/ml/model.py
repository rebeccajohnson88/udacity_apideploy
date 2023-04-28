from sklearn.metrics import fbeta_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # penalized logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # fit model
    lrc.fit(X_train, y_train)
    
    return lrc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : penalized logistic regression
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds 


def compute_metrics_slice(raw_data, one_feature,
                          y_test, preds):
    """ Computes performance metrics on a slice of the data

    Inputs
    ------
    raw_data: initial test split in raw pd form (not the np array)
    one_feature: categorical feature to slice by
    y_test: test set labels
    preds: test set predictions

    Returns
    -------
    writes a file slice_output that summarizes performance metrics
    for all district levels of a categorical variable
    """
    all_categories = raw_data[one_feature].unique()
    all_slices = [create_calc_subset(one_slice, one_feature, raw_data, y_test, preds) 
                  for one_slice in all_categories]
    slice_df = pd.concat(all_slices)
    with open("../slice_output.txt", 'w') as f:
        slice_string = slice_df.to_string(index=False)
        f.write(slice_string)

def create_calc_subset(one_slice, one_feature, test_df, y_test, preds):
    """ Helper function for compute_metrics_slice

    Inputs
    ------
    one_slice: one level of a categorical variable
    one_feature: name of categorical feature

    Returns
    -------
    stores a dataframe with the name of the categorical var,
    the name of the subgroup, and performance metrics for that 
    subgroup
    """
    store_results = {'feature': one_feature, 'subgroup': '', 
                'precision': '', 'recall': '', 'fbeta': ''}
    subset_df = test_df[one_feature] == one_slice
    subset_y = y_test[subset_df]
    subset_preds = preds[subset_df]
    precision, recall, fbeta = compute_model_metrics(subset_y, 
                                                subset_preds)
    store_results['precision'] = precision
    store_results['recall'] = recall
    store_results['fbeta'] = fbeta
    store_results['subgroup'] = one_slice
    return(pd.DataFrame(store_results, index = [0]))
