#importing all required libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#next, we will load the training dataset. 
#basically, we are using two dataset. one for training the model, and one for testing it.
data_path = "training.csv" #point the location of the dataset
data = pd.read_csv(data_path).dropna(axis=1) #check for presence of missing or null values, [axis=1 means "drop the entire column, not rows(individuals)"]


#next, we check for balance....we ask the question, is the dataset balance? we need to be sure all diseases are well-represented
disease_counts = data['prognosis'].value_counts()
temp_df = pd.DataFrame({
    "Disease":disease_counts.index,
    "Counts":disease_counts.values
})

#lets plot our result to see if we indeed have a balanced dataset

plt.figure(figsize= (18,5))
sns.barplot(x="Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90) # Rotate labels
plt.subplots_adjust(bottom=0.5)  # Adjust bottom margin (increase if needed)
plt.show()


#observe that the target column, prognosis, isnt a numerical value. to train our model, we need a numerical value, so we will be using 
#LabelEncoder from sklearn.preprocessing to encode the data, converting to a numerical value

#print(data.head())  # See original categorical "prognosis" column

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

##print(data.head())  # Now, "prognosis" contains numbers instead of disease names


#next thing we would wanna do is split the dataset into training and testing
x = data.iloc[:,:-1] #grab everything excluding the last column, here, we are extracting all the IV
y = data.iloc[:,-1]#grab the last column, which is the dependent variable, the target so to speak
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_train.shape}")


#next, we start building the model 
#we'll be using k-fold cross-validation for model selection

#defining scoring metrics for k-fold cross-validation
def CV_scoring(estimator,x,y):
    return accuracy_score(y, estimator.predict(x))

#initializing the models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

#producing cross-validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, x, y, cv = 10, n_jobs = -1, scoring = CV_scoring)
    print("=="*30)
    print(model_name)
    print(f"scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


#next, we would wanna try building a more robust classifier by combining all models
#we will train and test the model separately using each algorithm

#Training and testing SVM Classifier

svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)
print(f"Accuracy on train data by SVM Classifier\: {accuracy_score(y_train,svm_model.predict(X_train))*100}")
print(f"Accuracy on test data by SVM Classifier\: {accuracy_score(y_test,preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix,annot=True)
plt.title("confusion matrix for SVM classifier on Test data")
plt.show()

#Training and testing Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy score on train data by naive bayes classifier\: {accuracy_score(y_train,nb_model.predict(X_train))*100}")
print(f"Accuracy of test data by naive bayes classifier\: {accuracy_score(y_test,preds)}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix,annot=True)
plt.title("confusion matrix for naive bayes classifier on test data")
plt.show()

#Training and testing random forest classifier
r_forest = RandomForestClassifier(random_state=18)
r_forest.fit(X_train, y_train)
preds = r_forest.predict(X_test)
print(f"Accuracy score on train data by random forest classifier\: {accuracy_score(y_train, r_forest.predict(X_train))*100}")
print(f"Accuracy score on test data by random forest classifier\: {accuracy_score(y_test, preds)}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("confusion natrix for random forest classifier on test data")
plt.show()

#Now we will be training the models on the whole train data present in the dataset
#that we downloaded and then test our combined model on test data present in the dataset.

#Training the models on whole data

final_svm_model = SVC()
final_nb_model = GaussianNB()
final_r_forest_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(x,y)
final_nb_model.fit(x,y)
final_r_forest_model.fit(x,y)


#Reading the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)

test_x = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

#making prediction by take mode of predictions made by all the classifiers

svm_preds = final_svm_model.predict(test_x)
nb_preds = final_nb_model.predict(test_x)
r_forest_preds = final_r_forest_model.predict(test_x)

from scipy import stats

final_preds = [stats.mode([i,j,k])[0] for i,j,k in zip(svm_preds, nb_preds, r_forest_preds)]
print(f"Accuracy on test dataset by the combined model: {accuracy_score(test_y,final_preds)*100}")
cf_matrix = confusion_matrix(test_y, final_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("confusion matrix for combined model on test dataset")
plt.show() 


#next, we will be creating a function that can take symptoms as input and generate predictions for disease
symptoms = x.columns.values

#creating a symptom index dictionary to encode the input symptoms into numerical form
# Creating a symptom index dictionary to encode input symptoms into numerical form
symptom_index = {symptom.replace("_", " ").title(): i for i, symptom in enumerate(symptoms)}

# Storing data dictionary for predictions
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the prediction function
def predictDisease(symptoms_input):
    symptoms = symptoms_input.split(", ")
    
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    
    for symptom in symptoms:
        formatted_symptom = symptom.title()  # Ensure matching format
        if formatted_symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][formatted_symptom]
            input_data[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized. Please check spelling.")

    # Reshaping input data
    input_data = np.array(input_data).reshape(1, -1)

    # Generating individual predictions
    r_forest_prediction = data_dict["predictions_classes"][final_r_forest_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Making final prediction using mode
    import statistics
    final_prediction = statistics.mode([r_forest_prediction, nb_prediction, svm_prediction])

    predictions = {
        "Random Forest Prediction": r_forest_prediction,
        "Naive Bayes Prediction": nb_prediction,
        "SVM Prediction": svm_prediction,
        "Final Prediction": final_prediction
    }

    return predictions

# Testing the function
print(predictDisease("Itching, Skin Rash, Nodal Skin Eruptions"))
