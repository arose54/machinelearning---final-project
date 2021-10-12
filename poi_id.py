#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def listbycriteria(dictionary,feature,dvalue,op):
    keylist=[]
    for k,v in dictionary.items():
        if v[feature] != "NaN":
            if op == "=":
                if v[feature] == dvalue:
                    keylist.append(k)
                    print v[feature]
            elif op == "!=":
                if v[feature] != dvalue:
                    keylist.append(k)
                    print v[feature]
            elif op == ">":
                if v[feature] > dvalue:
                    keylist.append(k)
                    print v[feature] 
            elif op == "<":
                if v[feature] < dvalue:
                 keylist.append(k)
                 print v[feature]
    return keylist

def deleteNaNs(dictionary,feature):
    
    for k,v in dictionary.items():
        if v[feature] == "NaN":
            del dictionary[k]
            
    return dictionary

def plot_features(data_to_plot,features,feat_num1,feat_num2,lab1,lab2,col):
    for point in data_to_plot:
        
        if point[col]==0.0:
            color='orchid'
        else:
            color='mediumseagreen'
            
        feature_x = point[feat_num1]
        feature_y = point[feat_num2]
        plot=plt.scatter(feature_x,feature_y,c=color)
       
    plt.xlabel(lab1)
    plt.ylabel(lab2)
    plt.show()
    return plot

def poicountbyfeature(dictionary,feature,dvalue,op,boolPOI):
    poicount=0
    for k,v in dictionary.items():
        if op == "equal":
            if v[feature] == dvalue and v['poi'] == boolPOI:
                poicount += 1
        elif op == "not equal":
            if v[feature] != dvalue and v['poi'] == boolPOI: 
                poicount += 1
    return poicount

def datacount(dictionary,feature,dvalue,op):
    datacount=0
    for k,v in dictionary.items():
        if op == "equal":
            if v[feature] == dvalue:
                datacount += 1
        elif op == "not equal":
            if v[feature] != dvalue:
                datacount += 1
    return datacount

def checkfornans(dictionary,features):
    allstats=""
    nans=""
    poinans=""
    nonpoinans=""
    
    for feature in features:
        
        if feature != 'poi':
            nans = feature+": "+str(datacount(dictionary,feature,'NaN','equal'))+" NaNs and "+str(datacount(dictionary,feature,0,'equal'))+" zeros \n"
            poinans = str(poicountbyfeature(dictionary,feature,'NaN','equal',True))+" NaNs are for pois\n"
            nonpoinans = str(poicountbyfeature(dictionary,feature,'NaN','equal',False))+" are for non-pois\n\n"
            allstats = allstats+nans+poinans+nonpoinans
    
    return allstats
           

def strip_unwanted_keys(dictionary,key_value):
    ### Core of function adapted from https://stackabuse.com/python-how-to-remove-a-key-from-a-dictionary/
    return {k: v for k, v in dictionary.items() if k != key_value}

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
   
### In order to assess best features to use - I need to remind myself of
### the features available to me. I used method suggested in the following
### article, but revised to just iterate once, since the features are all the same.
### https://stackoverflow.com/questions/39233973/get-all-keys-of-a-nested-dictionary    
    featcntr = 0
    for k, v in data_dict.items():
        if featcntr == 0:
            for k1, v1 in v.items():
                print(k1)
        featcntr += 1
            
### In the list of features, I see some metrics related to email volume - I 
### think this may be useful for the "birds of a feather" factor,  but
### I would like to contextualize them in relation to each users overall volume
### to ensure we're not over-weighting folks who are just prolific users of email.
### I am also interested in measures of how many emails individuals send and receive 
### from POIs. The measures that exist are counts - I feel percentages will be better. 
### I will caculate and append new features to my dataset.
        
for k,v in data_dict.items():
        
        if v['from_poi_to_this_person'] == 'NaN':
            v.update(perc_from_poi=0)
        else:
            v.update(perc_from_poi=(float(v['from_poi_to_this_person'])/float(v['to_messages'])))
        if v['from_this_person_to_poi'] == 'NaN':
            v.update(perc_to_poi=0)
        else:
            v.update(perc_to_poi=(float(v['from_this_person_to_poi'])/float(v['from_messages'])))

        if v['from_this_person_to_poi'] == 'NaN':
            v.update(perc_shared_receipt=0)
        else:
            v.update(perc_shared_receipt=(float(v['shared_receipt_with_poi'])/float(v['to_messages'])))
     
### Now I will add these two new features to my features_list  
features_list.append('perc_from_poi')
features_list.append('perc_to_poi')
features_list.append('perc_shared_receipt')

### This third feature did not appear to have much value in a test run. Eliminating.
### features_list.append('perc_shared_receipt')
  

#### Task 2: Remove outliers
##### Where are the outliers?
###### Let's look at the "birds of a feather" measures first.
print features_list 
data_clean_1 = featureFormat(data_dict, features_list)

### Let's plot the data.
plot_features(data_clean_1,features_list,1,2,"perc_from_poi","perc_to_poi",0)

### One seems very high - let's see what that value is.
print listbycriteria(data_dict,'perc_to_poi',0.8,">")

### Wow - 1.0 - seems a bit high - does it make sense by the numbers? Yes, it does. 
print data_dict['HUMPHREY GENE E']

###### Now let's look at some financial figures. Let's add 'salary' and 'bonus' to our data set.
features_list.append('salary')
features_list.append('bonus')


data_clean_2 = featureFormat(data_dict, features_list, sort_keys = False)
plot_features(data_clean_2,features_list,4,5,"salary","bonus",0)

####Who is the person in the top right corner that looks like an outlier? 
print listbycriteria(data_dict,'salary',2500000,">")

#### 'TOTAL' is not legit data. I need to remove it. 
data_dict_wo_total = strip_unwanted_keys(data_dict,'TOTAL')

#### Let's see how it looks now. 
data_clean_3 = featureFormat(data_dict_wo_total, features_list, sort_keys = False)
plot_features(data_clean_3,features_list,4,5,"salary","bonus",0)

###Do the top few entries in each of these categories look legit?
print listbycriteria(data_dict_wo_total,'salary',1000000,">")
print listbycriteria(data_dict_wo_total,'bonus',5000000,">")

#### Now lets look at some other features
##### Total Payments & Stock Value
features_list.append('total_payments')
features_list.append('total_stock_value')

data_clean_4 = featureFormat(data_dict_wo_total, features_list)
plot_features(data_clean_4,features_list,6,7,"total_payments","total_stock_value",0)

#### What is this top right outlier? Well, Ken Lay - that's a legit point.
print listbycriteria(data_dict_wo_total,'total_payments',100000000,">")
print listbycriteria(data_dict_wo_total,'total_stock_value',12000000,">")

#### These really seem to be clustered at the lower end of the graph. Are there a lot of NaNs?
#### The feature format will switch NaNs to 0s- but if but if there are too many, it may compromise the value of the feature.
#print checkfornans(data_dict_wo_total,features_list)

#### When I run test algorithms on just the email/"birds of a feather" features
#### the metrics are pretty good.  But when I add the financial features, the
#### the recall and precision get very good for non-pois, but really low for 
#### pois.  Given that the the NaNs in these data sets are largely for non-pois,
#### it may be that the algorithm is interpreting NaNs/0s as indicators of non-pois.
#### What if I strip the records with financial values that are 'Nan' - will it improve?

deleteNaNs(data_dict_wo_total,'salary')
deleteNaNs(data_dict_wo_total,'bonus')
deleteNaNs(data_dict_wo_total,'total_payments')
deleteNaNs(data_dict_wo_total,'total_stock_value')

#### It did!

my_dataset = data_dict_wo_total 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap=False, class_weight='balanced',
            criterion='entropy', max_depth=None, max_features='auto',
            max_leaf_nodes=15, min_impurity_split=1e-07,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.3, n_estimators=5, n_jobs=5,
            oob_score=False, random_state=16, verbose=0, warm_start=False)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
