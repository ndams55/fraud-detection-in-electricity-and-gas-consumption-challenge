"""In this document, I explain how I do Feature Engineering for the dataset"""

# required librairies
import pandas as pd


def convert_to_date_time(dataset):
    """
        Convert the column invoice_date to date time format on both the invoice train and invoice test
    """
    for df in dataset:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])

    return df


def encode_label(df):
    """
        Encode labels in categorical column
    """
    d={"ELEC":0,"GAZ":1}
    df['counter_type']=df['counter_type'].map(d)

    return df
    


def categorical_to_int(df):
    """
        Convert categorical columns to int for model
    """
    df['client_catg'] = df['client_catg'].astype(int)
    return df



def aggregate_by_client_id(df):
    aggs = {}
    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']

    agg_trans = df.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (df.groupby('client_id')
            .size()
            .reset_index(name='{}transactions_count'.format('1')))
    
    return pd.merge(df, agg_trans, on='client_id', how='left')



def merge_data(invoice_data, client_data):
    """
        Merge aggregate data with client dataset
    """
    agg_data = aggregate_by_client_id(invoice_data)  #group invoice data by client_id
    data = pd.merge(client_data,agg_data, on='client_id', how='left')
    return data



def drop_redundant_col(train, test):
   """
    Drop redundant columns
   """
   drop_columns = ['client_id', 'creation_date']

   for col in drop_columns:
        if col in train.columns:
            train.drop([col], axis=1, inplace=True)
        if col in test.columns:
            test.drop([col], axis=1, inplace=True)
   return train, test


