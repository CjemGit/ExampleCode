#Script Using Python and Scikit Learn to classify userss based on their first booking location
#Demonstrates some basic feature engineering techniques and using a random forest classifier
#Data taken from Kaggle @ https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt import qgrid as q
from ggplot import *

##DATA IMPORT

test = pd.read_csv(’test_users.csv’)
train = pd.read_csv(’train_users_2.csv’)
labels = pd.DataFrame(train[’country_destination’].values)
train = train.drop([’country_destination’], axis=1)
total = pd.concat((test, train), axis=0, ignore_index=True)
testids = test[’id’]
total.drop(’id’,axis=1, inplace=True)

for column in total:
    if total[column].dtype==’object’:
        total[str(column)].replace(’NaN’, ’MISSING’, inplace=True)
        total[’signup_flow’] = total[’signup_flow’].astype(’object’)

##FEATURE ENGINEERING

#TIME
times = pd.DataFrame(total[[’timestamp_first_active’]])
times.columns = [’time’]
timesno = pd.DataFrame(pd.to_numeric(times[’time’], errors = ’
coerce’))
timesno = pd.DataFrame(timesno[’time’].astype(str).str[8:10]. astype(np.int64))
cuts = [-1,5,12,17,20,23]
cutnames = [’Late Night’,’Morning’,’Afternoon’,’Evening’,’Night’
]
timecuts = pd.cut(timesno[’time’], cuts, labels = cutnames)
total = pd.concat([total, timecuts], axis=1, join_axes=[total.
index])
season = pd.DataFrame(total[[’timestamp_first_active’]])
season.columns = [’season’]
seasonno = pd.DataFrame(pd.to_numeric(season[’season’], errors =
’coerce’))
seasonno = pd.DataFrame(seasonno[’season’].astype(str).str[4:6]. astype(np.int64))
cutnames2 = [’January’,’February’,’March’,’April’,’May’,’June’,’
July’,’August’,’September’,’October’,’November’,’December’]
seasoncuts = pd.cut(seasonno[’season’], 12, labels = cutnames2)
total = pd.concat([total, seasoncuts], axis=1, join_axes=[total.
index])

#STRANGE AGE
age = pd.DataFrame(total[’age’])
age = age.fillna(1)
StrangeAge = pd.DataFrame(age[(age>70)| (age<16)])
StrangeAge.columns = [’StrangeAge’]
StrangeAge[StrangeAge>0] = True
StrangeAge = StrangeAge.fillna(False)
StrangeAge[’StrangeAge’] = StrangeAge[’StrangeAge’].astype(’bool
’)
StrangeAge[’StrangeAge’].describe()
age[(age>70) | (age<16)] = np.NaN

#TRACKED
tracked = pd.DataFrame(total[’first_affiliate_tracked’])
tracked.columns= [’Tracked’]
untracked = pd.DataFrame(tracked.Tracked==’untracked’)
untracked.columns= [’Untracked’]

#AGE BINS
agebins = pd.cut(age[’age’],12)
total = pd.concat([total,agebins], axis=1, join_axes=[total.
index])
dummies = pd.DataFrame(total.iloc
[:,[3,5,6,7,8,9,11,12,13,14,15,16]])
dummydf = pd.get_dummies(dummies)
total2 = pd.concat([StrangeAge, untracked, dummydf, labels],
axis=1, join_axes=[total.index])
#total2 = pd.concat([StrangeAge, untracked, dummydf, labels],
axis=1, join_axes=[total.index])
final = pd.DataFrame(total2.iloc[:,16:])

#SPLITTING TRAIN AND TEST
piv_train = train.shape[0]
vals = final.values
X = vals[:piv_train]

from sklearn.model_selection import train_test_split
train, test = train_test_split(X, test_size = 0.2)
labels = train[:,161:]
testlabels = test[:,161:]
labels = np.ravel(labels)
testlabels = np.ravel(testlabels)
train2 = train[:,:161]
test2 = test[:,:161]

##CLASSIFIERS
from sklearn.ensemble import RandomForestClassifier

criteria = [’gini’,’entropy’]
for y in criteria:
    for n in range(5,30,5):
        rf = RandomForestClassifier(n_estimators=n, criterion=
        y, verbose=True)
        rf.fit(train2, labels)
        rfpredict = rf.predict(test2)
        unique = np.unique(testlabels)
        from sklearn.metrics import classification_report
        y_true = testlabels
        target_names = unique
        print(classification_report(y_true, rfpredict, target_names=target_names))
        print(’CRITIERA = ENTROPY’)
        print(classification_report(y_true, neighpredict, target_names=
        target_names))
