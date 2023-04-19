import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import fn




# Initialize the sweep
sweep_id = wandb.sweep(sweep_config)

# Run the sweep with train_and_evaluate function and datasets as arguments
