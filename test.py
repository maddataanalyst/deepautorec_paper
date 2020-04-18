import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


columns = [
    "acc_status",
    "duration",
    "history",
    "purpose",
    "amount",
    "savings",
    "employment_since",
    "installment_rate",
    "gender",
    "debtors",
    "residence_since",
    "property",
    "age",
    "installment_plans",
    "housing",
    "number_of_accounts",
    "job",
    "no_of_liable",
    "telephone",
    "foreign_worker",
    "cls"
]
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv", header=None)
data.columns = columns

transformer = ColumnTransformer([
    #('le', LabelEncoder(), [0])
    ('ohe', OneHotEncoder(), [0])
], remainder='passthrough')

data_fit = transformer.fit_transform(data)
