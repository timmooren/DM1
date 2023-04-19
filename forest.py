from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import functions as fn
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def main():
    data = fn.clean_data()
    data = data
    X = data.drop(columns=['time', 'mood', 'mood_new'])
    y = data['mood_new']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred)) # gives precision, recall and F1

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    # save
    plt.savefig('plots/cm.png')

    first_tree = random_forest.estimators_[0]
    # Set up the figure size
    plt.figure(figsize=(20, 10))
    # Plot the tree
    plot_tree(first_tree, feature_names=X_train.columns, class_names=True, filled=True)
    # save
    plt.savefig('plots/tree.png')
