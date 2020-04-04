import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('creditcard.csv')
data.sample(frac=0.1)     # samples only 10% of the data for developmental use


def preprocessor():
    """
    returns features and labels - training and testing data, and ratio of fraud to total transactions (contamination)

    for this project,   the features would be data - the "Class" column
                        the labels would be the "Class" column
    :return:
    """
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    ratio = len(fraud) / (len(fraud) + len(valid))

    columns = data.columns.to_list()  # list of all columns
    columns = [c for c in columns if c != "Class"]  # list of all columns except "Class"

    features = data[columns]
    labels = data["Class"]

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.3)

    return features_train, features_test, labels_train, labels_test
