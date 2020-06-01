# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
features = ['Pclass', 'Sex',  'SibSp', 'Parch']
features_train = pd.get_dummies(train_data[features])     #importing our data provided by kaggle
labels_train = train_data['Survived']
features_test = pd.get_dummies(test_data[features])
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=6, algorithm='brute')   
classifier.fit(features_train, labels_train)  # trainning our model

pred = classifier.predict(features_test)  # predicitions

#converting to a .csv file to submit
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
output.to_csv('Submission.csv', index=False)
print('done')
