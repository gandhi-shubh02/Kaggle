import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a
# version using "Save & Run All" You can also write temporary files to /kaggle/temp/, but they won't be saved outside
# of the current session 

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') # obtaining our data
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
features = ['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotalBsmtSF']

from sklearn.impute import SimpleImputer

imputer = SimpleImputer() # using an imputer to replace missing data with mean
labels_train = train_data['SalePrice']
features_train = pd.get_dummies(train_data[features])
features_test = pd.get_dummies(test_data[features])
features_train = imputer.fit_transform(features_train)
features_test = imputer.fit_transform(features_test)

from sklearn.linear_model import LinearRegression
                                                                        # Using a basic Linear model
clf = LinearRegression().fit(features_train, labels_train)
predprice = clf.predict(features_test)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predprice})
output.to_csv('submission.csv', index=False)
print('done')
