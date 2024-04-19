# %% [markdown]
# <img src="https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6)" alt="tree" width=1510px height=410px>
# 

# %% [markdown]
# #  <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Cancer Diagnosis with Various Classifiers</span>

# %% [markdown]
# <a href="https://www.linkedin.com/in/okinoleiba" style="">Okino Kamali Leiba</a>

# %%
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# %%
data_set = "C:/Users/Owner/source/vsc_repo/confusion_matrix_cookbook/diagnosis_confusion_matrix/breast-cancer_diagnostic.csv"
cancer_data = pd.read_csv(data_set, delimiter=",", encoding="utf-8", header=0, on_bad_lines="warn")

# %%
import sqlite3

file_location = "C:/Users/Owner/source/vsc_repo/etl_pipeline-world bank/datasource/population_data.db"
file_type = "db"

connection = sqlite3.connect(file_location, uri = True)

pop_data_sqlite = pd.read_sql("select * from population_data", connection)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;"> Exploratory Data Analysis</span>

# %%
cancer_data.index

# %%
cancer_data.columns

# %%
cancer_data.info()

# %%
cancer_data.dtypes

# %%
cancer_data.head(5)

# %%
cancer_data.tail(5)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Data Transformation and Preparation</span>

# %%
cancer_data.drop(["id","Unnamed: 32"], axis= 1, errors="ignore", inplace=True)

# %%

# for diagnosis in cancer_data.diagnosis:
#     if diagnosis == "M":
#          cancer_data.diagnosis = 1
#     else:
#         cancer_data.diagnosis = 0
# cancer_data.diagnosis=[1 if each == "M" else 0 for each in cancer_data.diagnosis]
# cancer_data.diagnosis = [diagnosis = 1 for diagnosis in cancer_data.diagnosis if diagnosis == "M" else 0 ]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cancer_data["diagnosis"] = le.fit_transform(cancer_data["diagnosis"])
cancer_data.head(3)

# %%
features = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
X = cancer_data[features]
# X = cancer_data[0:,1:20]
# X = cancer_data.drop('diagnosis', axis=1, inplace=False, errors="ignore")
# y = cancer_data.diagnosis.values
y = cancer_data["diagnosis"]


# %%
from sklearn.preprocessing import MinMaxScaler
X = (X - np.min(X)) / (np.max(X) - np.min(X))
mms = MinMaxScaler(X)


# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: black;">Model Generation with PyCaret</span>

# %%
from pycaret.classification import *
cls = setup(data=cancer_data, target="diagnosis")

# %%
cm = compare_models()
# et	Extra Trees Classifier
# xgboost	Extreme Gradient Boosting

# %%
crtm = create_model("et")


# %%
interpret_model(crtm)

# %%
plot_model(cm, plot='confusion_matrix', plot_kwargs={"percent":True})

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Gradient Boosting Classifier</span>

# %%
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=1)
gbc.fit(X_train, y_train)
print("Gradient Boosting Classifier: ", gbc.score(X_test, y_test))

# %%
y_predict = gbc.predict(X_test)

cm_gbc = confusion_matrix(y_test, y_predict)
cm_gbc

# %%
# Accuracy: (True Positive + True Negative) / Total Predictions
# Precision: True Positive / (True Positive + False Positive)
# Sensitivity: True Positive / (True Positive + False Negative)
# Specifity: True Negative / (True Negative + False Positive)
# F-Score: 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
true_negative, false_positive, false_negative, true_positive  = cm_gbc.ravel()
true_negative, false_positive, false_negative, true_positive

# %%
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
sensitivity = true_positive / (true_positive + false_negative)
specifity = true_negative / (true_negative + false_positive)
f_score = 2 * ((precision * sensitivity)) / ((precision + sensitivity))
print("Accuracy_Extra Tree Classifier: ", accuracy)
print("Precision_Extra Tree Classifier: ", precision)
print("Sensitivity_Extra Tree Classifier: ", sensitivity)
print("Specifity_Extra Tree Classifier: ", specifity)
print("F-Score_Extra Tree Classifier: ", f_score)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Extra Trees Classifier</span>

# %%
from sklearn.tree import ExtraTreeClassifier
et = ExtraTreeClassifier(random_state=1)
X_train_et = X_train.copy()
X_test_et = X_test.copy()
y_train_et = y_train.copy()
y_test_et = y_test.copy()
et.fit(X_train_et, y_train_et)
print("Extra Trees Classifier: ", et.score(X_test_et, y_test_et))

# %%
y_predict_et = et.predict(X_test_et)

cm_et = confusion_matrix(y_test_et, y_predict_et)
cm_et

# %%
# Accuracy: (True Positive + True Negative) / Total Predictions
# Precision: True Positive / (True Positive + False Positive)
# Sensitivity: True Positive / (True Positive + False Negative)
# Specifity: True Negative / (True Negative + False Positive)
# F-Score: 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
true_negative, false_positive, false_negative, true_positive  = cm_et.ravel()
true_negative, false_positive, false_negative, true_positive

# %%
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
sensitivity = true_positive / (true_positive + false_negative)
specifity = true_negative / (true_negative + false_positive)
f_score = 2 * ((precision * sensitivity)) / ((precision + sensitivity))
print("Accuracy_Extra Tree Classifier: ", accuracy)
print("Precision_Extra Tree Classifier: ", precision)
print("Sensitivity_Extra Tree Classifier: ", sensitivity)
print("Specifity_Extra Tree Classifier: ", specifity)
print("F-Score_Extra Tree Classifier: ", f_score)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Random Forest Classifier</span>

# %%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
y_train_rf = y_train.copy()
y_test_rf = y_test.copy()
rf.fit(X_train_rf, y_train_rf)
print("Random Forest Classifier: ", rf.score(X_test_rf, y_test_rf))

# %%
y_predict_rf = rf.predict(X_test_rf)

cm_rf = confusion_matrix(y_test_rf, y_predict_rf)
cm_rf


# %%
# Accuracy: (True Positive + True Negative) / Total Predictions
# Precision: True Positive / (True Positive + False Positive)
# Sensitivity: True Positive / (True Positive + False Negative)
# Specifity: True Negative / (True Negative + False Positive)
# F-Score: 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
true_negative, false_positive, false_negative, true_positive  = cm_rf.ravel()
true_negative, false_positive, false_negative, true_positive

# %%
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
sensitivity = true_positive / (true_positive + false_negative)
specifity = true_negative / (true_negative + false_positive)
f_score = 2 * ((precision * sensitivity)) / ((precision + sensitivity))
print("Accuracy_Random Forest Classifier: ", accuracy)
print("Precision_Random Forest Classifier: ", precision)
print("Sensitivity_Random Forest Classifier: ", sensitivity)
print("Specifity_Random Forest Classifier: ", specifity)
print("F-Score_Random Forest Classifier: ", f_score)

# %%
cm_plt = pd.DataFrame([cm_rf[0], cm_rf[1]], index=pd.Index(["Non-Cancer(Negative)", "Cancer(Positive"], name="Actual Label: "), columns=pd.MultiIndex.from_product([["Confusion Matrix of Cancer Diagnosis"], ["Non_Cancer(False)", "Cancer(True)"]], names=[" ", "Predicted: "]))
cm_plt.style

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Adaptive Boost Classifier</span>

# %%
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=100, random_state=1)
X_train_abc = X_train.copy()
X_test_abc = X_test.copy()
y_train_abc = y_train.copy()
y_test_abc = y_test.copy()
abc.fit(X_train_abc, y_train_abc)
print("Adaptive Boost Classifier: ", abc.score(X_test_abc, y_test_abc))

# %%
y_predict_abc = abc.predict(X_test_abc)

cm_abc = confusion_matrix(y_test_abc, y_predict_abc)
cm_abc

# %%
# Accuracy: (True Positive + True Negative) / Total Predictions
# Precision: True Positive / (True Positive + False Positive)
# Sensitivity: True Positive / (True Positive + False Negative)
# Specificity: True Negative / (True Negative + False Positive)
# F-Score: 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
true_negative, false_positive, false_negative, true_positive  = cm_abc.ravel()
true_negative, false_positive, false_negative, true_positive

# %%
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
f_score = 2 * ((precision * sensitivity)) / ((precision + sensitivity))
print("Accuracy_Adaptive Boost Classifier: ", accuracy)
print("Precision_Adaptive Boost Classifier: ", precision)
print("Sensitivity_Adaptive Boost Classifier: ", sensitivity)
print("Specificity_Adaptive Boost Classifier: ", specificity)
print("F-Score_Adaptive Boost Classifier: ", f_score)

# %%
plt.figure(figsize=(8,6), constrained_layout=False, dpi=100, facecolor="orange")
axe = sns.heatmap(cm_abc, annot=True,  fmt=".3g", cbar= False, xticklabels=["False", "True"], yticklabels=["False", "True"]);
axe.set_xlabel("Predicted");
axe.set_ylabel("Actual");


# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">K-Nearest Neighbors Classifier</span>

# %%
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="brute")
X_train_knc = X_train.copy()
X_test_knc = X_test.copy()
y_train_knc = y_train.copy()
y_test_knc = y_test.copy()
knc.fit(X_train_knc, y_train_knc)
print("K Neighbors Classifier: ", knc.score(X_test_knc, y_test_knc, sample_weight=None))

# %%
y_predict_knc = knc.predict(X_test_knc)

cm_knc = confusion_matrix(y_test_knc, y_predict_knc)
cm_knc

# %%
# Accuracy: (True Positive + True Negative) / Total Predictions
# Precision: True Positive / (True Positive + False Positive)
# Sensitivity: True Positive / (True Positive + False Negative)
# Specificity: True Negative / (True Negative + False Positive)
# F-Score: 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
true_negative, false_positive, false_negative, true_positive  = cm_knc.ravel()
true_negative, false_positive, false_negative, true_positive

# %%
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
f_score = 2 * ((precision * sensitivity)) / ((precision + sensitivity))
print("Accuracy_K Neighbors Classifier: ", accuracy)
print("Precision_K Neighbor Classifier: ", precision)
print("Sensitivity_K Neighbor Classifier: ", sensitivity)
print("Specificity_K Neighbor Classifier: ", specificity)
print("F-Score_K Neighbor Classifier: ", f_score)

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">AUC - ROC</span>

# %%
plot_model(cm, plot='auc')

# %%

y_predict_rf = rf.predict(X_test)
y_predict_abc = abc.predict(X_test)
y_predict_knc = knc.predict(X_test)
y_predict = [y_predict_rf, y_predict_abc, y_predict_knc]
for y_proba in y_predict:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr,label="AUC="+str(auc))
    plt.legend()
    plt.xlabel("Recall: False Positive Rate")
    plt.ylabel("Precision: True Positive Rate")

# %% [markdown]
# ## <span style="background-image: url(https://img.freepik.com/free-vector/hand-drawn-tree-life-brown-shades_23-2148703761.jpg?w=740&t=st=1683924116~exp=1683924716~hmac=61b5f87828f15616fb97b47c8068990abdee0657d540607c983bc4e586c862e6); background-size: cover; background-repeat: no-repeat; color: #4d4d4d;">Mean Squared Error</span>

# %%
from sklearn.metrics import mean_squared_error
print("Gradient Boosting Classifier: ", mean_squared_error(y_test, gbc.predict(X_test)))
print("Extra Trees Classifier: ", mean_squared_error(y_test, et.predict(X_test)))
print("Random Forest Classifier: ", mean_squared_error(y_test, rf.predict(X_test)))
print("Adaptive Boost Classifier: ", mean_squared_error(y_test, abc.predict(X_test)))
print("K-Nearest Neighbors Classifier: ", mean_squared_error(y_test, knc.predict(X_test)))


