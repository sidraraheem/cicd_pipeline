# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Clean feature names
dataset.columns = [colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]

# Identify and encode the target column
if 'target' not in dataset.columns:
    # Most iris datasets have 'species' or 'variety' column
    possible_targets = [col for col in dataset.columns if 'species' in col.lower() or 'class' in col.lower() or 'variety' in col.lower()]
    if len(possible_targets) == 0:
        raise ValueError("❌ No target column found in iris.csv (expected 'species' or 'class').")
    target_col = possible_targets[0]
    le = LabelEncoder()
    dataset['target'] = le.fit_transform(dataset[target_col])
else:
    le = None

# Feature Engineering
dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Select Features
dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                   'sepal_length_width_ratio', 'petal_length_width_ratio', 'target']]

# Split Data (stratified to avoid single-class issue)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44, stratify=dataset['target'])

# Prepare train/test sets
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')
X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')

# Logistic Regression
logreg = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=200, multi_class='multinomial')
logreg.fit(X_train, y_train)
predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)
f1_lr = f1_score(y_test, predictions_lr, average='micro')
prec_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')

# Accuracy
train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# Random Forest (Classifier instead of Regressor)
rf_clf = RandomForestClassifier(random_state=44)
rf_clf.fit(X_train, y_train)
predictions_rf = rf_clf.predict(X_test)

f1_rf = f1_score(y_test, predictions_rf, average='micro')
prec_rf = precision_score(y_test, predictions_rf, average='micro')
recall_rf = recall_score(y_test, predictions_rf, average='micro')

# Accuracy
train_acc_rf = rf_clf.score(X_train, y_train) * 100
test_acc_rf = rf_clf.score(X_test, y_test) * 100

# Confusion Matrix Plotting Function
def plot_cm(cm, target_names, title="Confusion Matrix", normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:0.3f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel(f"Predicted Label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    plt.savefig("ConfusionMatrix.png", dpi=120)
    plt.close()

# Plot Confusion Matrix
if le:
    target_names = le.classes_
else:
    target_names = np.unique(y_train)
plot_cm(cm_lr, target_names, title="Confusion Matrix (Logistic Regression)")

# Feature Importance Plot
importances = rf_clf.feature_importances_
labels = dataset.columns[:-1]
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=['feature', 'importance'])
features = feature_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=features)
plt.title("Random Forest Feature Importances", fontsize=14)
plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=120)
plt.close()

# Save Results
with open("scores.txt", "w") as score:
    score.write(f"Random Forest Train Var: {train_acc_rf:.2f}%\n")
    score.write(f"Random Forest Test Var: {test_acc_rf:.2f}%\n")
    score.write(f"F1 Score: {f1_rf:.2f}%\n")
    score.write(f"Recall Score: {recall_rf:.2f}%\n")
    score.write(f"Precision Score: {prec_rf:.2f}%\n\n")

    score.write(f"Logistic Regression Train Var: {train_acc_lr:.2f}%\n")
    score.write(f"Logistic Regression Test Var: {test_acc_lr:.2f}%\n")
    score.write(f"F1 Score: {f1_lr:.2f}%\n")
    score.write(f"Recall Score: {recall_lr:.2f}%\n")
    score.write(f"Precision Score: {prec_lr:.2f}%\n")

print("✅ Training completed successfully. Scores and plots saved.")





