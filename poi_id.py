#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier



# -------------------------
# STEP 0
# Preparations
# -------------------------

import csv
import pandas
import pprint
from time import time
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Quick prints of key data
print "\n ############## Stats on dataset ############\n"
print "Number of people in dataset: %d" %len(data_dict.keys())

POIs = sum ( 1 if data_dict[key]['poi']==True else 0 for key in data_dict)
print "\nNumber of POIs:", POIs

print "\nNumber of features available:", len(list(data_dict["SKILLING JEFFREY K"].keys()))

print "\nListing of features available:"
pprint.pprint(list(enumerate(data_dict["SKILLING JEFFREY K"].keys())))

print "\n ############## Stats on NaNs ############\n"
  
### export dictionary to a CSV file for exploration, compute the fraction of NaNs
with open('data_dict.csv', 'wb') as csvfile:
    header = ["PERSON"]
    fields = data_dict[data_dict.keys()[0]].keys()
    header.extend(fields)
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=header)
    nanalysis = pandas.DataFrame(columns=["nans","values"], index=fields).fillna(0.0)
    for person in data_dict:
        row = {'PERSON': person}
        row.update(data_dict[person])
        for field in data_dict[person]:
            if data_dict[person][field] == 'NaN':
                nanalysis["nans"][field] += 1.0
            nanalysis["values"][field] += 1.0
        writer.writerow(row)
    nanalysis["frac"] = nanalysis["nans"]/nanalysis["values"]
    print "Proportion of NaNs per feature:\n"
    pprint.pprint(nanalysis.sort("frac", ascending=False)["frac"])
    print

"""
Proportion of NaNs per feature:

loan_advances                0.972603
director_fees                0.883562
restricted_stock_deferred    0.876712
deferral_payments            0.732877
deferred_income              0.664384
long_term_incentive          0.547945
bonus                        0.438356
from_poi_to_this_person      0.410959
shared_receipt_with_poi      0.410959
from_this_person_to_poi      0.410959
to_messages                  0.410959
from_messages                0.410959
other                        0.363014
salary                       0.349315
expenses                     0.349315
exercised_stock_options      0.301370
restricted_stock             0.246575
email_address                0.239726
total_payments               0.143836
total_stock_value            0.136986
poi                          0.000000
Name: frac, dtype: float64
"""



# -------------------------
# STEP 1 (Task 2)
# Remove outliers
# -------------------------

# Two outliers identified as selection errors
data_dict.pop("TOTAL", None)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)



# -------------------------
# STEP 2 (Task 1)
# Indicate what features you may use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# -------------------------

"""
Listing of features available:
[(0, 'salary'),
 (1, 'to_messages'),
 (2, 'deferral_payments'),
 (3, 'total_payments'),
 (4, 'exercised_stock_options'),
 (5, 'bonus'),
 (6, 'restricted_stock'),
 (7, 'shared_receipt_with_poi'),
 (8, 'restricted_stock_deferred'),
 (9, 'total_stock_value'),
 (10, 'expenses'),
 (11, 'loan_advances'),
 (12, 'from_messages'),
 (13, 'other'),
 (14, 'from_this_person_to_poi'),
 (15, 'poi'),
 (16, 'director_fees'),
 (17, 'deferred_income'),
 (18, 'long_term_incentive'),
 (19, 'email_address'),
 (20, 'from_poi_to_this_person')]
"""

### The first feature must be "poi".
features_list = ['poi']

# financial features, minus 'loan_advances', 'director_fees',  'restricted_stock_deferred'
financial_features=['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                     'bonus', 'restricted_stock', 'total_stock_value', 'expenses', 'other',
                     'deferred_income', 'long_term_incentive']


# email_features
# removed 'email_adress': no use in this case of ML, like person's name
email_features=['to_messages', 'shared_receipt_with_poi', 'from_this_person_to_poi', 
                    'from_poi_to_this_person']



# -------------------------
# STEP 3 (Task 3)
# Create new feature(s)
# Store to my_dataset for easy export below.
# -------------------------

## 3.1 New features from the available dataset

## Email features: creation of three ratios to reflect relative values instead of absolute.
## - 'from_poi_to_this_person_ratio'
## - 'from_this_person_to_poi_ratio'
##- 'shared_receipt_with_poi_ratio'

for person in data_dict.keys():
    all_to=data_dict[person]['to_messages']
    all_from=data_dict[person]['from_messages']
    to_poi=data_dict[person]['from_this_person_to_poi']
    from_poi=data_dict[person]['from_poi_to_this_person']
    cc_with_poi=data_dict[person]['shared_receipt_with_poi']
    if all_to=='NaN' or from_poi=='NaN':
        data_dict[person]['from_poi_to_this_person_ratio']='NaN'
    else:
        data_dict[person]['from_poi_to_this_person_ratio']=1.0*from_poi/all_to
    if all_from=='NaN' or to_poi=='NaN':
        data_dict[person]['from_this_person_to_poi_ratio']='NaN'
    else:
        data_dict[person]['from_this_person_to_poi_ratio']=1.0*to_poi/all_from
    if all_to=='NaN' or cc_with_poi=='NaN':
        data_dict[person]['shared_receipt_with_poi_ratio']='NaN'
    else:
        data_dict[person]['shared_receipt_with_poi_ratio']=1.0*cc_with_poi/all_to


## Financial features: creation of a 'payment_&_stock' feature

for person in data_dict.keys():
    pay_n_stock_features = ['salary', 'bonus', 'other', 'total_stock_value']
    pay_n_stock = 0
    for feature in pay_n_stock_features:
        if data_dict[person][feature] != 'NaN':
            pay_n_stock += data_dict[person][feature]
    data_dict[person]['pay_n_stock'] = pay_n_stock

## Available print checks to verify addition of new features to dataset
#print "\nNumber of features available after new features update:", len(list(data_dict["SKILLING JEFFREY K"].keys()))
#print "\nListing of features available after new features update:"
#pprint.pprint(list(enumerate(data_dict["SKILLING JEFFREY K"].keys())))


# Updated financial features
financial_features_2=['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                     'bonus', 'restricted_stock', 'total_stock_value', 'expenses', 'other',
                     'deferred_income', 'long_term_incentive', 'pay_n_stock']


# Updated email_features
email_features_2=['to_messages', 'shared_receipt_with_poi', 'from_this_person_to_poi', 
                    'from_poi_to_this_person', 'from_poi_to_this_person_ratio',
                    'from_this_person_to_poi_ratio', 'shared_receipt_with_poi_ratio']


my_dataset = data_dict
all_features = features_list + financial_features_2 + email_features_2

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# K-best features - choosing up to 10 features for a trial
k_best = SelectKBest(f_classif, k=10)
k_best.fit(features, labels)

result_list = zip(k_best.get_support(), all_features[1:], k_best.scores_)
result_list = sorted(result_list, key=lambda x: x[2], reverse=True)
print "\nK-best features - i.e. top 10 features selected:"
pprint.pprint(result_list)


#### FEATURES EXPLORATION ZONE WITH RAW RESULTS FROM FULL RUN #####

features_1 = features_list + ['pay_n_stock']
"""
NB
Precision: 0.48959
Recall: 0.22350
DT
Precision: 0.39786
Recall: 0.33400
RF
Precision: 0.39268
Recall: 0.28450
"""

features_3 = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus']
"""
3
NB
Precision: 0.48581
Recall: 0.35100
DT
Precision: 0.36615
Recall: 0.39050
RF
Precision: 0.60138
Recall: 0.30550
"""

features_3_bis = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'pay_n_stock']
"""
3_bis
NB
Precision: 0.46686
Recall: 0.32400
DT
Precision: 0.36686
Recall: 0.40850
RF
Precision: 0.52519
Recall: 0.31800
"""

features_4 = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'salary']
"""
NB
Precision: 0.50312
Recall: 0.32300
DT
Precision: 0.31870
Recall: 0.33400
RF
Precision: 0.48646
Recall: 0.23350
"""

features_4_bis = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'salary',
                              'pay_n_stock',]
"""
NB
Precision: 0.47050
Recall: 0.32300
DT
Precision: 0.36377
Recall: 0.37050
RF
Precision: 0.51819
Recall: 0.26350
"""

features_5 = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'salary',
                              'from_this_person_to_poi_ratio']
"""
NB
Precision: 0.49545
Recall: 0.32650
DT
Precision: 0.29953
Recall: 0.34850
RF
Precision: 0.45825
Recall: 0.21950
"""

features_5_bis = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'pay_n_stock',
                              'salary',
                              'from_this_person_to_poi_ratio']
"""
NB
Precision: 0.46193
Recall: 0.31550
DT
Precision: 0.31465
Recall: 0.36200
RF
Precision: 0.43837
Recall: 0.20450
"""

features_6= features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'salary',
                              'from_poi_to_this_person_ratio',
                              'deferred_income']
"""
NB
Precision: 0.51572
Recall: 0.38550
DT
Precision: 0.26833
Recall: 0.25800
RF
Precision: 0.47820
Recall: 0.18100
"""

features_6_bis = features_list + ['exercised_stock_options',
                              'total_stock_value',
                              'bonus',
                              'pay_n_stock',
                              'salary',
                              'from_poi_to_this_person_ratio',
                              'deferred_income']
"""
NB
Precision: 0.48573
Recall: 0.39150
DT
Precision: 0.30716
Recall: 0.28750
RF
Precision: 0.46650
Recall: 0.18450
"""

#### Selection of features_set for full run
my_features = features_3_bis

#### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)



# -------------------------
# STEP 4 (Task 4):
# Try a varity of classifiers
# -------------------------

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Feature Scaling applied
features = StandardScaler().fit_transform(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels)


# create standard classifier function
def clf(type):
    t0 = time()
    clf = type()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print '+++ One fold score +++'
    print "Accuracy_score:", accuracy_score(pred, labels_test)
    print 'Precision:', precision_score(pred, labels_test)
    print 'Recall:', recall_score(pred, labels_test)
    print 'Time:',round(time()-t0,3) ,'s\n'
    print '+++ 1000 folds score +++'
    test_classifier(clf, my_dataset, my_features)


#### START FULL RUN
print '\nSTARTING FULL RUN WITH:\n'
print '\nMy_features selected:'
pprint.pprint(my_features, width = 5)

print "\n***** DEFAULT TRAIN/TEST SPLIT *****"

print '\n----- NB classifier -----'
clf(GaussianNB)

print '\n----- SVM classifier -----'
clf(SVC)

print '\n----- Decision Tree classifier -----'
clf(tree.DecisionTreeClassifier)

print '\n----- Random Forest classifier -----'
clf(RandomForestClassifier)

print '\n----- Adaboost classifier -----'
clf(AdaBoostClassifier)

print '\n----- K nearest Neighbor classifier -----'
clf(KNeighborsClassifier)

print '\n----- Logistic Regression classifier -----'
clf(LogisticRegression)



# -------------------------
# STEP 5 (Task 5):
# Tune your classifier to achieve better
# than .3 precision and recall 
# -------------------------

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# STEP 5.1 :
##### Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "\n***** TEST_SIZE = 0.3 SPLIT *****"

print '\n----- NB classifier -----'
clf(GaussianNB)

print '\n----- Decision Tree classifier -----'
clf(tree.DecisionTreeClassifier)

print '\n----- Random Forest classifier -----'
clf(RandomForestClassifier)


# STEP 5.1 :
##### Implement GridSearchCV for Random Forest

#http://scikit-learn.org/0.17/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py

print "\nImplementing GridSearchCV for Random Forest"

# train_test_split reversed to default test_size=0.25
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels)

# Set the parameters by cross-validation
tuned_parameters = {'n_estimators': [20,50,100,200],
                    'min_samples_split': [2,3,4],
                    'max_features': [2,3,4]}
                    
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    GridSearch = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    GridSearch.fit(features_train, labels_train)

    print("\nBest parameters set found on development set:\n")
    print(GridSearch.best_params_)


    print("\nDetailed classification report:\n")
    labels_true, labels_pred = labels_test, GridSearch.predict(features_test)
    print(classification_report(labels_true, labels_pred))

### FINAL CLASSIFIER

print '+++ RECALL RECOMMENDATION +++'
clf = RandomForestClassifier(max_features= 3, min_samples_split= 3, n_estimators= 100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
test_classifier(clf, my_dataset, my_features)



# -------------------------
# STEP 6 (Task 6):
#`Dump your classifier
# -------------------------

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features)
