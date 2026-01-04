#required libraries for reading dataset and plotting the graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#machine learning
from sklearn.model_selection import train_test_split #pip install scikit-learn
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data_path = "students.csv" 
df = pd.read_csv(data_path)
print(df.head()) #first 5 rows
print(df.info()) #information about entire dataset --> information of column and datatype of each column ex: studentID (number), name (text)
print("summary statistics")
print(df.describe(include="all")) #information and summarizing the records, == comparision operator , = assigning operator, mean, mode, standard deviation

#clean the duplicate values
df = df.drop_duplicates() #drop the duplicate values

#replace records with mean
numeric_cols = df.select_dtypes(include = [np.number]).columns #choosing specific columns, include numerical columns
for col in numeric_cols:
    df[col]=df[col].fillna(df[col].mean()) #all the empty values and fill it with the mean values of the empty areas

#print most reoccuring value for each column
#for col in df.columns:
   # print(f'column:{col}')
  #  print(df[col].mode().iloc[0]) 

#print most reoccuring value for name
print(df["Name"].mode().iloc[0])



#replace NaN records with mode for category columns (gender)
categoric_cols = df.select_dtypes(include = ["object"]).columns
for cat in categoric_cols:
    df[cat] = df[cat].fillna(df[cat].mode()[0])
print(df.head())

#check for missing values after cleaning information
print(df.isna().sum()) #isna = to check missing values
df = pd.read_csv(data_path)

#average final score by agenda
avg_by_gender = df.groupby("Gender")["FinalScore"].mean()
print("\n Average Final Score By Gender")
print(avg_by_gender)

#group records according to two columns
#avg_by_gender = df.groupby("Gender, ______class")["FinalScore"].mean()
#print("\n Average Final Score By Gender")
# print(avg_by_gender)

#plot a graph
plt.figure(figsize=(6, 4))
plt.title("Comparision between FinalScore and Count")
sns.histplot(df["FinalScore"],kde=True, color = "red", edgecolor="green", linewidth=1.3) #kde = visualization method to estimate probablity 
plt.show()

#corelation heatmaps for numeric columns
plt.figure(figsize=(8, 6))
corr=df.select_dtypes(include = [np.number]).corr()
sns.heatmap(corr, annot=True, cmap="plasma")
plt.show()

#correlation between final score and study time
plt.figure(figsize=(8, 6))
#corr=df["StudyTime"].corr(df["FinalScore"])
cols = ["StudyTime", "FinalScore"]
corr=df[cols].corr()
sns.heatmap(corr, annot=True, cmap="plasma")
plt.show()

#correlation and heatmap for numeric columns (HOMEWORK-tell about that; continue)
#add some more records; minimum 10-15 records; figsize, kde, creation of csv files

feature_cols = ["StudyTime", "Attendance", "PreviousScore", "Gender"]
threshold = 75
df['PerformanceLabel']=np.where(df['FinalScore']>=threshold, 'High', 'Low')
target_col = "PerformanceLabel"
df_model = pd.get_dummies(df[feature_cols], drop_first=True)

#encode performance label: high = 1, low = 0
y = np.where(df[target_col]=="High", 1, 0)
x = df_model.values
print('Feature Columns After Encoding')
print(df_model.columns)

#train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#standarize numeric features
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(X_train)
x_test_scale = scaler.transform(X_test)

#predict information - logistical regressions (classification algorithm which predits probabilty of an event occuring [0, 1])

#logistic regression
log_reg = LogisticRegression(max_iter=2000) #builds the logistic regression model
log_reg.fit(x_train_scale, y_train) #trains it on the scaled training data
y_pred_lr = log_reg.predict(x_test_scale) #predicts the output for the test data
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr,
       labels=[0, 1],
       target_names=["Low", "High"]))
print(np.unique(y_test), np.unique(y_pred_lr))

#confusion matrix
print(confusion_matrix(y_test, y_pred_lr))

#random forest classifier
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred_rf,
       labels=[0, 1],
       target_names=["Low", "High"]))

#feature importance plot
importance = rf_clf.feature_importances_ #get importance score for each feature
feature_names = df_model.columns 
feature_imp = pd.Series(importance, index = feature_names).sort_values(ascending = False) #pd.Series = matches score with the feature names, and then it will sort the data

plt.figure(figsize = (6, 4))
sns.barplot(x = feature_imp.values, y = feature_imp.index)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

#save the processed data
df_results = df.copy()
df_results["predicted_performance_rf"] = np.where(rf_clf.predict(df_model.values)==1, "High", "Low") #creating another column after checking if the prediction is high/low
output_path = "students_results_predicted.csv" #put two additional columns in the dataset: PerformanceLabel, predicted_performance_rf
df_results.to_csv(output_path, index = False)
#print("Results saved to", output_path)
print(f"Results saved to: {output_path}")
