import re
import string
from nltk import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.translate import metrics
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm,datasets


#TODO: Reading and Displaying Datasets

# Set display options to show all columns
pd.set_option('display.max_columns', None)

# Read datasets and handle their text in Latin
Questions = pd.read_csv('Questions.csv', encoding='latin')
Answers = pd.read_csv('Answers.csv', encoding='latin')
Tags = pd.read_csv('Tags.csv', encoding='latin')

print(Questions.head(3))
print("############################################################################################################################################")
print(Answers.head(3))
print("############################################################################################################################################")
print(Tags.head(3))
# Check the data types of each column
#data_types_tags= Tags.dtypes
#print(data_types_tags)


#############################################################################################################################
#TODO:Dataset Preparation

#rename columns names of question and Answer
Questions.columns=['Id','OwnerUserId',	'CreationDate',	'CloseDate' , 'Score' , 'Title' , 'Question']#body to Question
Answers.columns=['Id_normal', 'OwnerUserId', 'CreationDate','Id','Score','Answer']#parentId to Id/body to Answer


# Dropping unecessary columns
Answers.drop(columns=['Id_normal', 'OwnerUserId', 'CreationDate'], inplace=True)

#grouping the answers based on the 'Id' column and then joining the individual answers within each group into a single string
Answers = Answers.groupby('Id')['Answer'].apply(lambda answer: ' '.join(answer))
Answers = Answers.to_frame().reset_index()

# Changing the data type of 'Tag' column from object to string
Tags['Tag']= Tags['Tag'].astype(str)

# Joining tags grouped by 'Id'
Tags = Tags.groupby('Id')['Tag'].apply(lambda tag: ' '.join(tag))
Tags = Tags.to_frame().reset_index()

#Merging all dataset to a Single dataset
new_data = Questions.merge(Answers, how='left', on='Id')
new_data = new_data.merge(Tags, how='left', on='Id')

# Dropping unecessary columns
new_data.drop(columns=[ 'OwnerUserId' , 'CreationDate' ,	'CloseDate' ], inplace=True)

#rename columns names of new data
new_data.columns = ['id','score','title','question','answer','tag']

# Creating 'tagcount' column,counts the occurrences of each tag
count = new_data.groupby('tag')['tag'].count()
count = count.to_frame()

#rename column name
count.columns = ['TagCount']
count = count.reset_index()

# Merging created column to the existing dataframe
new_data = pd.merge(new_data, count, how='left', on='tag')

#check null values
null_values = new_data.isnull().sum()
print("####################################################null values###################################################################")
print(null_values)#answer

new_data = new_data.dropna()#drop null values

#note that : for better accuracy may can drop answer column which had null values  -_-

print(new_data.shape)
##reduce data
new_data = new_data[(new_data['TagCount'] >= 1100) & (new_data['score'] > 7)]


print("################################################Data After Preparation################################################################ ")
print(new_data)



new_data.drop(columns=['score', 'id','TagCount'], inplace=True)
######################################################################################################################################
#TODO:Dataset Preprocessing

Lematizer = WordNetLemmatizer()
# Defining a function to remove punctuation
def punctuation_remover(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


#Defining a lemmatizer function
def Word_Lemmatizer(text):
    lemma = [Lematizer.lemmatize(word) for word in text]
    return lemma


# Changing the data type of 'title' , 'answer' and 'question' columns to string
new_data['title'] = new_data['title'].astype(str)
new_data['question'] = new_data['question'].astype(str)
new_data['answer'] = new_data['answer'].astype(str)


# Applying 'punctuation_remover' function on 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(punctuation_remover)
new_data['question'] = new_data['question'].apply(punctuation_remover)
new_data['answer'] = new_data['answer'].apply(punctuation_remover)
print("##########################################################Data after punctuation_remover############################################### ")
print(new_data)

# Changing texts into lowercase
new_data['title'] = new_data['title'].str.lower()
new_data['question'] = new_data['question'].str.lower()
new_data['answer'] = new_data['answer'].str.lower()

# Removing HTML tags on 'title' , 'answer' and 'question' columns
new_data['question'] = new_data['question'].apply(lambda question: re.sub('<[^<]+?>', '', question))
new_data['answer'] = new_data['answer'].apply(lambda answer: re.sub('<[^<]+?>', '', answer))
new_data['title'] = new_data['title'].apply(lambda title: re.sub('<[^<]+?>', '', title))
print("#############################################Data after Removing HTML tags and Changing texts into lowercase ############################# ")
print(new_data)

# Splitting the texts into words (segmantation)
new_data['question'] = new_data['question'].str.split()
new_data['answer'] = new_data['answer'].str.split()
new_data['title'] = new_data['title'].str.split()

# Applying lemmatizer function to 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(lambda title: Word_Lemmatizer(title))
new_data['answer'] = new_data['answer'].apply(lambda answer: Word_Lemmatizer(answer))
new_data['question'] = new_data['question'].apply(lambda question: Word_Lemmatizer(question))
print("###################################################Data after Applying lemmatizer######################################################## ")
print(new_data)
# Removing Stopword from 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(lambda title: [word for word in title if word not in stopwords.words('english')])
new_data['question'] = new_data['question'].apply(lambda question: [word for word in question if word not in stopwords.words('english')])
new_data['answer'] = new_data['answer'].apply(lambda answer: [word for word in answer if word not in stopwords.words('english')])
print("######################################################Data after Removing Stopword######################################################### ")
#print(new_data)


##########################################################################################################################################
#TODO: Features extraction(TF-IDF Vectorization):-

vectorizer = TfidfVectorizer()

# Changing the data type of 'title' ,'answer' and 'question' columns to string
new_data['title'] = new_data['title'].astype(str)
new_data['answer'] = new_data['answer'].astype(str)
new_data['question'] = new_data['question'].astype(str)

X1 = vectorizer.fit_transform(new_data['title'].str.lower())
X2 = vectorizer.fit_transform(new_data['answer'].str.lower())
X3 =vectorizer.fit_transform(new_data['question'].str.lower())

# Initialize LabelEncoder
label_encoder = LabelEncoder()


new_data['tag'] = label_encoder.fit_transform(new_data['tag'])
y = new_data['tag'].values


##########################################################################################################################################
# TODO: Model training and testing:-

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X2, new_data['tag'], test_size=0.4, random_state=10)
x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(X2, new_data['tag'], test_size=0.35, random_state=10)

# Define models
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': svm.SVC(kernel='linear', C=10, random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=2000),
    'Decision Tree': DecisionTreeClassifier(random_state=10),
    'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=10),
    'Logistic Regression': LogisticRegression()
}

# Train and test models
accuracies = {}

for model_name, model in models.items():
    if model_name == 'KNN':
        accuracy = []
        for i in range(1, 100):
            knn = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
            prediction = knn.predict(x_test)
            accuracy.append(accuracy_score(y_test, prediction))
        accuracies[model_name] = max(accuracy)
    elif model_name == 'SVM':
        svm_model = svm.SVC(kernel='linear', C=10, random_state=0).fit(x_train_svm, y_train_svm)
        pred_svm = svm_model.predict(x_test_svm)
        accuracies[model_name] = accuracy_score(y_test_svm, pred_svm)
    else:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracies[model_name] = accuracy_score(y_test, pred)

# Print accuracies
for model_name, accuracy in accuracies.items():
    print(f'Accuracy of {model_name}: {accuracy}')
##########################################################################################################################################
#TODO: Results visualization:-


# Visualization of KNN
plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), accuracies['KNN'] * 99, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()


# Visualization of SVM
plt.figure(figsize=(10, 6))
plt.bar(['SVM'], [accuracies['SVM']], color='orange')
plt.title('SVM Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Visualization of Random Forest
plt.figure(figsize=(10, 6))
plt.bar(['Random Forest'], [accuracies['Random Forest']], color='purple')
plt.title('Random Forest Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Visualization of Decision Tree
plt.figure(figsize=(10, 6))
plt.bar(['Decision Tree'], [accuracies['Decision Tree']], color='yellow')
plt.title('Decision Tree Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Visualization of GBM
plt.figure(figsize=(10, 6))
plt.bar(['GBM'], [accuracies['GBM']], color='green')
plt.title('GBM Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Visualization of Logistic Regression
plt.figure(figsize=(10, 6))
plt.bar(['Logistic Regression'], [accuracies['Logistic Regression']], color='magenta')
plt.title('Logistic Regression Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()