import pandas as pd
import numpy as np


file_location = "heart.csv"
data = pd.read_csv(file_location)
label = 'target'

cols = data.columns.tolist()
colIdx = data.columns.get_loc(label)

if colIdx != 0:
    cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]

modified_data = data[cols]
MAX_CAT_ALLOWED = 10
cat_cols = modified_data.select_dtypes(exclude=['int', 'float']).columns
cat_cols = set(cat_cols) - {label}

useless_cols = []
for cat_column_features in cat_cols:
    num_cat = modified_data[cat_column_features].nunique()
    if num_cat > MAX_CAT_ALLOWED:
        useless_cols.append(cat_column_features)

for feature_column in modified_data.columns:
    num_cat = modified_data[feature_column].nunique()
    if num_cat <= 1:
        useless_cols.append(feature_column)
modified_data = modified_data.drop(useless_cols, axis=1)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data_without_label = modified_data.drop([label], axis=1)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = data_without_label.select_dtypes(include=['int64',
                                                    'float64']).columns

categorical_features = data_without_label.select_dtypes(exclude=['int64',
                                                            'float64']).columns

preprocessor_cols = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])

preprocessor = Pipeline(steps=[('preprocessor', preprocessor_cols)])
preprocessor.fit(data_without_label)
modified_data_without_label = preprocessor.transform(data_without_label)
if (type(modified_data_without_label) is not np.ndarray):
    modified_data_without_label = modified_data_without_label.toarray()

modified_data_array = np.concatenate(
    (np.array(modified_data[label]).reshape(-1, 1),
     modified_data_without_label), axis=1)
np.savetxt("data_processed.csv", modified_data_array, delimiter=",", fmt='%1.3f')

from sklearn.model_selection import train_test_split
train, test= train_test_split(modified_data_array, test_size=0.2)
np.savetxt("train.csv", train, delimiter=",", fmt='%1.3f')
np.savetxt("test.csv", test, delimiter=",", fmt='%1.3f')