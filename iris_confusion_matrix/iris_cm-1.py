# %% [markdown]
# # <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Iris Classifier by Various Models</span>

# %% [markdown]
# <a href="https://www.linkedin.com/in/okinoleiba" style="">Okino Kamali Leiba</a>

# %%
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os

# %%
data_set = "./iris.csv"
iris_data = pd.read_csv(data_set, engine="c", delimiter=",", encoding="utf-8", header=0, on_bad_lines="warn")
iris_data["Species"].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], ['Iris-Setosa','Iris-Versicolor', 'Iris-Virginica'], regex=True, inplace=True)

# %%
from io import StringIO
python_data = open(data_set).read()
lst_com = [list_item.split(",") for list_item in python_data.splitlines()]
# data_clip = pd.read_clipboard(python_data)
# data_table = pd.read_table(python_data)
data_csv = pd.read_csv(StringIO(python_data), header=0, sep=",", engine="c", lineterminator="\n", iterator=True, chunksize=100)

# https://matthewrocklin.com/blog/work/2017/10/16/streaming-dataframes-1

pd.DataFrame(lst_com)

# %% [markdown]
# ## <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Exploratory Data Analysis</span>

# %%
iris_data.index

# %%
iris_data.columns

# %%
iris_data.dtypes

# %%
iris_data.info()

# %%
iris_data["Species"].unique()

# %%
iris_data.head(5)

# %%
iris_data.tail(5)

# %% [markdown]
# ## <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Set Figure Theme</span>

# %%
custom_params = {'figure.facecolor': 'orange',
 'axes.labelcolor': '.15',
 'xtick.direction': 'out',
 'ytick.direction': 'out',
 'xtick.color': '.15',
 'ytick.color': '.15',
 'axes.axisbelow': True,
 'grid.linestyle': '-',
 'text.color': '.15',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'DejaVu Sans',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'image.cmap': 'rocket',
 'xtick.top': True,
 'ytick.right': True,
 'axes.grid': True,
 'axes.facecolor': '#EAEAF2',
 'axes.edgecolor': 'black',
 'grid.color': 'white',
 'axes.spines.left': False,
 'axes.spines.bottom': False,
 'axes.spines.right': False,
 'axes.spines.top': False,
 'xtick.bottom': True,
 'ytick.left': True}
sns.set_theme(style="ticks", rc=custom_params)

# %% [markdown]
# ## <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Data Transformation and Preparation</span>

# %%
iris_data = iris_data.drop("Id", axis=1, errors="ignore", inplace=False)

# %%

iris_data.plot.hist(subplots=True, figsize=(12,6),);

# %%
x = iris_data.drop("Species", axis=1, inplace=False, errors="ignore")
# X scale-min_max = (X — X min) / (X max — X min)
X = (x - np.min(x)) / (np.max(x) -np.min(x))
# X scale_MAS = x / max(abs|x|)
# X = x / np.max(np.abs(x))
# X z-score = (x - mean / (x - std)  *normal distribution
#   X = (x - np.mean(x)) / (x - np.std(x))
y = iris_data["Species"]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09, random_state=0)

# %% [markdown]
# ## <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Random Forest Classifier</span>

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

param_grid = {"n_estimators" : [50, 100, 150, 200], "max_depth" : [2, 4, 6, 8, 10], "min_samples_split" : [2, 4, 6, 8, 10], "min_samples_leaf" : [2, 4, 6, 8, 10], "max_features" : ["sqrt", 
"log2"], "random_state" : [0,42]}
search_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5, scoring = "f1_samples" )
search_grid.fit(X_train, y_train)
print(search_grid.best_params_)

rf = RandomForestClassifier(n_estimators=50, max_depth=2, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, random_state=0)
rf.fit(X_train, y_train)
print("Random Forest Classifier: ", rf.score(X_test, y_test))

# %%
from sklearn.metrics import confusion_matrix
y_predict_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_predict_rf)
cm_rf

# %%
from sklearn.metrics import ConfusionMatrixDisplay
fig, axe = plt.subplots(figsize=(10,4), constrained_layout=True, dpi=100)
# cm_display = ConfusionMatrixDisplay(cm_rf).plot()
cm_display = ConfusionMatrixDisplay(cm_rf)
cm_display.plot(ax=axe);

# %%
plt.figure(figsize=(10,4), constrained_layout=True, dpi=100)
axe = sns.heatmap(cm_rf, annot=True, fmt=".3g", cbar= False, xticklabels=["Setosa","Versicolor","Virginica"], yticklabels=["Setosa","Versicolor","Virginica"]);
axe.set_xlabel("Predicted");
axe.set_ylabel("Actual");


# %% [markdown]
# ## <span style="background-color: #dda0dd; color: black; border-left: 15px solid #FFD700;">Support Vector Classifier</span>

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {"kernel" : ["linear", "poly", "rbf", "sigmoid"], "degree" : [3, 6, 9, 12, 15], "random_state" : [0, 42]  }
grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

svc = SVC(kernel="linear", random_state=0, degree=3)
svc.fit(X_train, y_train)
print("Support Vector Classifier: ", svc.score(X_test, y_test))

# %%
from sklearn.metrics import confusion_matrix
y_predict_svc = svc.predict(X_test)

cm_svc = confusion_matrix(y_test, y_predict_svc)
cm_svc

# %%
plt.figure(figsize=(5,3), constrained_layout=True, dpi=100)
sns.clustermap(cm_svc,  xticklabels=["Setosa","Versicolor","Virginica"], yticklabels=["Setosa","Versicolor","Virginica"], center=0, cmap="viridis", annot=True, dendrogram_ratio=(.1, .2), cbar_pos=(.02, .32, .03, .2), linewidths=.75)
plt.show()


# %%
from pandas.plotting import lag_plot

plt.figure(figsize=(10,4), constrained_layout=True, dpi=100)

data = pd.Series(0.1 * np.random.rand(1000) + 0.9 * np.sin(np.linspace(-99 * np.pi, 99 * np.pi, num=1000)))

lag_plot(data)


