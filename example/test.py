from logic import convert_to_date_time, encode_label, categorical_to_int, merge_data, drop_redundant_col 
from logic import NeuralNet, FraudDetection
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import os.path
from os import path


if __name__ == "__main__":

    DATA_DIR = '../dataset'

    TRAIN_DIR = f'{DATA_DIR}/train'
    TEST_DIR = f'{DATA_DIR}/test'

    OUTPUT_DIR = f'{DATA_DIR}/output'

    ## Download and extract files"""



    for pth in [TRAIN_DIR, TEST_DIR, OUTPUT_DIR]:
        if path.exists(pth) == False:
            os.mkdir(pth)

    """# Data Prep

    """## Read the Data"""

    client_train = pd.read_csv(f'{TRAIN_DIR}/client_train.csv', low_memory=False)
    invoice_train = pd.read_csv(f'{TRAIN_DIR}/invoice_train.csv', low_memory=False)

    client_test = pd.read_csv(f'{TEST_DIR}/client_test.csv', low_memory=False)
    invoice_test = pd.read_csv(f'{TEST_DIR}/invoice_test.csv', low_memory=False)
    sample_submission = pd.read_csv(f'{DATA_DIR}/SampleSubmission.csv', low_memory=False)

    """## Feature Engineering"""

    invoice_train['invoice_date'] = convert_to_date_time(invoice_train['invoice_date'])
    invoice_test['invoice_date'] = convert_to_date_time(invoice_test['invoice_date'])


    invoice_train['counter_type']=encode_label(invoice_train['counter_type'])
    invoice_test['counter_type']=encode_label(invoice_test['counter_type'])

   
    client_train['client_catg'] = categorical_to_int(client_train['client_catg'].astype(int))
    client_train['disrict'] = categorical_to_int(client_train['disrict'].astype(int))

    client_test['client_catg'] = categorical_to_int(client_test['client_catg'].astype(int))
    client_test['disrict'] = categorical_to_int(client_test['disrict'].astype(int))


    train = merge_data(invoice_train, client_train)

    test = merge_data(invoice_test, client_test)

    train, test = drop_redundant_col(train, test)

    #save the preprocess train and test dataset
    # train.to_csv(f'{TRAIN_DIR}/preprocess_train.csv')

    # test.to_csv(f'{TEST_DIR}/preprocess_test.csv')



    model = NeuralNet(input_size = train.shape[1], hidden_size_layer1 = 64, hidden_size_layer2 = 128, output_size = 1)

    criter = nn.BCELoss()
    optimize = torch.optim.SGD(model.parameters(), lr=10e-5)

    model.set_parameters(n_epochs = 100, learning_rate = 10e-5, optimize, criter)

    model.fit(train)

    model.predict(test)







