from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_eval_classical(X_train, y_train, X_test, y_test, cfg):
    models = {
        "SVM": SVC(kernel=cfg["classical"]["svm_kernel"]),
        "k-NN": KNeighborsClassifier(n_neighbors=cfg["classical"]["knn_n_neighbors"]),
        "DecisionTree": DecisionTreeClassifier(),
        "NaiveBayes": GaussianNB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1": f1_score(y_test, y_pred, average="macro"),
            "ConfusionMatrix": confusion_matrix(y_test, y_pred),
            "model": model
        }
    return results
