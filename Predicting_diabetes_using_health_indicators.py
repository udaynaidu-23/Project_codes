# Load Packages import numpy as np import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns import plotly.express as px import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_absolute_percentag
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_scor

d_df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv") d_df.head()

d_df.shape

d_df.columns

d_df.info()

d_df.describe().T

d_df[d_df.duplicated()]

d_df.drop_duplicates(inplace=True) d_df.shape

d_df.hist(figsize=(20,15));

d_df.corr()

plt.figure(figsize = (20,10)) sns.heatmap(d_df.corr(),annot=True) plt.title("correlation of feature") plt.show()

d_df["Diabetes_binary_eda"]= d_df["Diabetes_binary"].replace({0:"Non-Dia d_df.head()"})

d_df["Diabetes_binary_eda"].value_counts()]

plt.pie(d_df["Diabetes_binary"].value_counts() , labels =["non-Diabetic"plt.show()])

labels=["non HighBP","HighBP"]

plt.pie(d_df["HighBP"].value_counts() , labels =labels ,autopct='%.02f'


pd.crosstab(d_df.HighBP,d_df.Diabetes_binary_eda)

pd.crosstab(d_df.HighBP,d_df.Diabetes_binary_eda).plot(kind="bar",figsiz

plt.title('Diabetes Disease Frequency for HighBP') plt.xlabel("HighBP")
plt.ylabel('Frequency') plt.show()


labels=["non HighChol","HighChol"] plt.pie(d_df["HighChol"].value_counts() , labels =labels, autopct='%.02f


pd.crosstab(d_df.HighChol,d_df.Diabetes_binary_eda).plot(kind="bar",figs

plt.title('Diabetes Disease Frequency for HighChol') plt.xlabel("HighChol")
plt.ylabel('Frequency') plt.show()

ax= px.treemap(d_df,path=['BMI'],title="BMI counts") ax.show()

d_df[d_df["Diabetes_binary"] == 0].BMI

sns.boxplot(d_df.BMI)

labels=["non Smoker","Smoker"]
plt.pie(d_df["Smoker"].value_counts() , labels =labels ,autopct='%.02f')

pd.crosstab(d_df.Smoker,d_df.Diabetes_binary_eda).plot(kind="bar",figsiz

plt.title('Diabetes Disease Frequency for Smoker') plt.xlabel("Smoker")
plt.ylabel('Frequency') plt.show()


labels=["non HvyAlcoholConsump","HvyAlcoholConsump"]
plt.pie(d_df["HvyAlcoholConsump"].value_counts() , labels =labels ,autop

pd.crosstab(d_df.HvyAlcoholConsump,d_df.Diabetes_binary_eda).plot(kind="

plt.title('Diabetes Disease Frequency for HvyAlcoholConsump') plt.xlabel("HvyAlcoholConsump")
plt.ylabel('Frequency') plt.show()

pd.crosstab(d_df.Age,d_df.Diabetes_binary).plot(kind="bar",figsize=(20,6 plt.title('Diabetes Disease Frequency for Ages')
plt.xlabel('Age') plt.xticks(rotation=0) plt.ylabel('Frequency') plt.show()

plt.figure(figsize=(10,6))

sns.distplot(d_df.Education[d_df.Diabetes_binary == 0], color="r", label sns.distplot(d_df.Education[d_df.Diabetes_binary == 1], color="g", label plt.title("Relation b/w Education and Diabetes")

plt.legend() plt.show()


d_df.drop('Diabetes_binary_eda', axis=1, inplace=True)
X = d_df.drop('Diabetes_binary', axis=1) y = d_df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,


rf = RandomForestClassifier(max_depth=13, criterion='gini', n_estimators rf.fit(X_train, y_train)


importances = rf.feature_importances_ feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_v feature_importances

y_pred=rf.predict(X_test)
print('Training set score: {:.4f}'.format(rf.score(X_train, y_train))) print('Test set score: {:.4f}'.format(rf.score(X_test, y_test)))

mse =mean_squared_error(y_test, y_pred) print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(y_test, y_pred)) print('Root Mean Squared Error : '+ str(rmse))

rf_matrix = classification_report(y_test,y_pred) print(rf_matrix)

cm_rf = confusion_matrix(y_test,y_pred) plot_confusion_matrix(conf_mat=cm_rf,show_absolute=True, show_normed=Tru plt.show()

lg=LogisticRegression(C=1.0, random_state=42) lg.fit(X_train,y_train)

y_pred=lg.predict(X_test)
print('Training set score: {:.4f}'.format(lg.score(X_train, y_train))) print('Test set score: {:.4f}'.format(lg.score(X_test, y_test)))

mse=mean_squared_error(y_test,y_pred) print('Mean Squared Error : '+str(mse)) rmse=math.sqrt(mse)
print('Root Mean Squared Error : '+str(rmse))

lg_matrix = classification_report(y_test,y_pred) print(lg_matrix)
                                                              
# calculating the confusion matrix
cm_lg = confusion_matrix(y_test,y_pred) plot_confusion_matrix(conf_mat=cm_lg,show_absolute=True,show_normed=True plt.show()



