# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
training_data = pd.read_csv("training_data.csv")
#training_data.to_html('templates/original.html')

training_data.drop(['id_str', 'screen_name', 
                    'location', 'description', 
                    'url', 'created_at', 
                    'lang', 'status',
                    'default_profile',
                    'default_profile_image',
                    'has_extended_profile','name'],axis=1,inplace=True)

#training_data.to_html('templates/cleaned.html')

#To check Performances
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#calculate accuracy score, precision, recall, f1 score

def Performance(actual_value , predicted_value):
    accuracy = accuracy_score(actual_value , predicted_value) * 100
    precision = precision_score(actual_value , predicted_value) * 100
    recall = recall_score(actual_value , predicted_value) * 100
    f1 = f1_score(actual_value , predicted_value, average='weighted')
    print('Accuracy is {:.4f}%\n Precision is {:.4f}%\n Recall is {:.4f}%\nF1 Score is {:.4f}\n'.format(accuracy, precision, recall, f1))
    return accuracy,precision,recall,f1

#Extracted features

# features = ['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'verified', 'statuses_count','bot']
#---------------------------------------KNN Algorithm--------------------------------------------------
X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, 7].values

from sklearn.neighbors import KNeighborsClassifier as knn

classifier=knn(n_neighbors=5)
classifier.fit(X,y)

bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,7]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,7]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)

KNN = Performance(NB_y,NB_pred)[0]


#---------------------SVM----------------------------------------
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X,y)


bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,7]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,7]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
SVM = Performance(NB_y,NB_pred)[0]


#-----------------------Random Forest ----------------------------------------
from sklearn.ensemble import RandomForestClassifier as rf
classifier= rf(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)


bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]

B = bots.iloc[:,:-1]
B_y = bots.iloc[:,7]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,7]
NB_pred = classifier.predict(NB)
 
#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
RandomForest = Performance(NB_y,NB_pred)[0]

#------------------------------Naive Bayes---------------------------------------

from sklearn.naive_bayes import GaussianNB as GNB
classifier=GNB()
classifier.fit(X,y)


bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,7]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,7]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
NB = Performance(NB_y,NB_pred)[0]

#---------------------------Decision Tree---------------------------------
from sklearn.tree import DecisionTreeClassifier as DTC
classifier= DTC(criterion="entropy")
dtcClassifier = classifier
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,7]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,7]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
DT = Performance(NB_y,NB_pred)[0]


#-------------------------------------Custom Algorithm-----------------------------------------
class twitter_bot(object):
    def __init__(self):
        pass

    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        X_train, y_train = train, train.iloc[:,-1]
        X_test, y_test = test, test.iloc[:, -1]
        return (X_train, y_train, X_test, y_test)

    def bot_prediction_algorithm(df):
        train_df = df.copy()
        # converting id to int
        train_df['id'] = train_df.id.apply(lambda x: int(x))
        bag_of_words_bot = r'BOT|Bot|bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                           r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                           r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                           r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'


        # converting verified into vectors
        train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)
        # check if the name contains bot or screenname contains b0t
        condition = ((train_df.name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.description.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.status.str.contains(bag_of_words_bot, case=False, na=False))
                     )  # these all are bots
        predicted_df = train_df[condition]  # these all are bots
        predicted_df.bot = 1
        predicted_df = predicted_df[['id', 'bot']]

                # check if the user is verified
        verified_df = train_df[~condition]
        condition = (verified_df.verified == 1)  # these all are nonbots
        predicted_df1 = verified_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if description contains buzzfeed
        buzzfeed_df = verified_df[~condition]
        condition = (buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False))  # these all are nonbots
        predicted_df1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False)][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if listed_count>16000
        listed_count_df = buzzfeed_df[~condition]
        listed_count_df.listedcount = listed_count_df.listedcount.apply(lambda x: 0 if x == 'None' else x)
        listed_count_df.listedcount = listed_count_df.listedcount.apply(lambda x: int(x))
        condition = (listed_count_df.listedcount > 16000)  # these all are nonbots
        predicted_df1 = listed_count_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])
        #remaining
        predicted_df1 = listed_count_df[~condition][['id', 'bot']]
        predicted_df1.bot = 0 # these all are nonbots
        predicted_df = pd.concat([predicted_df, predicted_df1])
        return predicted_df

    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)

    def get_accuracy_score(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # predictions on training data
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        return(Performance(y_pred_train, y_true_train))


train_ = pd.read_csv('training_data.csv')
test_ = pd.read_csv('testing_data.csv')

train_.drop(['id_str', 'location', 'url', 'created_at', 'lang', 'has_extended_profile'],axis=1,inplace=True)
test_.drop(['id_str', 'location', 'url', 'created_at', 'lang', 'has_extended_profile'],axis=1,inplace=True)  
predicted_df = twitter_bot.bot_prediction_algorithm(test_)   
Custom = twitter_bot.get_accuracy_score(train_)[0]

#-----------------------------------Best Algorithm----------------------------------------------------------
'''
acc = np.array([KNN,SVM,RandomForest,NB,DT,Custom])
print("acc", acc[3])
algo = ['KNN','SVM','Random Forest','Naive Bayes','Decision Tree','Cutom Algorithm']
idx = np.argsort(acc)

plt.bar(acc,algo)
plt.savefig('static/images/best.jpg')

best_algo = algo[idx[-1]]
'''

best_algo = 'Decision Tree'

def userPrediction(test_sample):
    result = dtcClassifier.predict(test_sample)
    return "Non Bot" if result == 0 else "Bot"