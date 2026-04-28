# IMPORTS
import os # Allows us to navigate folders & find files
import numpy as np # Handles number arrays & math
import pandas as pd # Organizes data into clean tables
import matplotlib.pyplot as plt # Draws charts & visualizations
import seaborn as sns # Better graphs, good for confusion matrices

# scikit-learn for Decision Tree & evaluation tools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier  # MLP = Neural Network
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.metrics import (
    accuracy_score,        # what % of predictions were correct
    precision_score,       # of predicted class X, how many were actually X
    recall_score,          # of actual class X, how many did we catch
    f1_score,              # balance between precision & recall
    classification_report, # prints all metrics at once
    confusion_matrix       # table showing what got misclassified
)
from sklearn.preprocessing import LabelEncoder  # converts tile letters to numbers
 
import torch  # PyTorch: confirms GPU is available

# CONFIGURATION
# paths to dataset folders
MARIO_PATH = "data/mario" # Nintendo's Super Mario Bros dataset
MEGAMAN_PATH = "data/megaman" # Capcom's Mega Man dataset
 
# how much data to hold back for testing [0.2 = 20%]
TEST_SIZE = 0.2
 
# makes results reproducible: same split every time program is ran to provide consistency for the related paper
RANDOM_STATE = 42
 
# How many tiles around each tile observed
NEIGHBORHOOD_SIZE = 1  # look 1 tile in each direction (3x3 grid)
 
# Save locations for all output figures & metrics
OUTPUT_FIGURES = "outputs/figures"
OUTPUT_METRICS = "outputs/metrics"

#  LOAD AND PARSE LEVEL FILES
def load_levels(folder_path):

    # Reads all .txt level files from a folder and returns a list where each item is one level [a 2D grid of characters]
    levels = []  # holds all levels as a 2D list
 
    # loops through every file in the folder
    for filename in os.listdir(folder_path):
 
        # only process .txt files, skip anything else
        if filename.endswith(".txt"):
 
            filepath = os.path.join(folder_path, filename)
 
            # opens & reads file
            with open(filepath, "r") as f:
                lines = f.readlines()
 
            # each line = one row, strip removes the newline character at the end
            level = [list(line.strip()) for line in lines if line.strip()]
 
            # only adds levels w/ content
            if level:
                levels.append(level)
 
    print(f"Loaded {len(levels)} levels from {folder_path}")
    return levels

# PREPROCESSING: CONVERT ASCII TO FEATURES (X) AND LABELS (Y)
def encode_tiles(levels):
    
    # Converts all tile characters across all levels into integers and returns the encoder and encoded levels.
    
    # collects every unique tile character that appears
    all_chars = set()
    for level in levels:
        for row in level:
            for tile in row:
                all_chars.add(tile)

    # LabelEncoder maps each unique character to a unique integer
    encoder = LabelEncoder()
    encoder.fit(list(all_chars))  # teaches it all possible tile types

    # encodes every tile in every level
    encoded_levels = []
    for level in levels:
        encoded = []
        for row in level:
            encoded.append(encoder.transform(row))

        # find the longest row & pad shorter rows to match
        max_len = max(len(row) for row in encoded)
        padded = np.array([
            np.pad(row, (0, max_len - len(row)), constant_values=0)
            for row in encoded
        ])
        encoded_levels.append(padded)

    print(f"Tile types found: {list(encoder.classes_)}")
    print(f"Encoded as integers: {list(range(len(encoder.classes_)))}")
    return encoder, encoded_levels
 
 
def extract_features(encoded_levels, neighborhood=NEIGHBORHOOD_SIZE):
    # Extracts neighborhood feature vectors (X) and tile labels (Y) from all encoded levels.
  
    X = []  # feature vectors: one per tile
    Y = []  # labels: the tile type at the center
 
    for level in encoded_levels:
        rows, cols = level.shape  # get level dimensions
 
        # skip edge tiles because their neighborhood is incomplete
        # start at 'neighborhood' offset from each edge
        for r in range(neighborhood, rows - neighborhood):
            for c in range(neighborhood, cols - neighborhood):
 
                # extract the neighborhood around tile (r, c) | slices a (2n+1) x (2n+1) subgrid 
                patch = level[
                    r - neighborhood: r + neighborhood + 1,
                    c - neighborhood: c + neighborhood + 1
                ]
 
                # flatten 2D patch into 1D feature vector
                X.append(patch.flatten())
 
                # label = the tile type at the center of the patch
                Y.append(level[r, c])
 
    X = np.array(X)  # converts list of arrays to one big matrix
    Y = np.array(Y)  # converts list of labels to one array
 
    print(f"Total samples extracted: {X.shape[0]}")
    print(f"Feature vector size: {X.shape[1]} (neighborhood tiles)")
    return X, Y

# DATA VISUALIZATION
def visualize_level(level, title="Game Level", encoder=None):
    """
    Draws a color-coded visualization of one level.
    Each tile type gets a different color & Saves the figure to outputs/figures/
    """
    plt.figure(figsize=(20, 5))  # wider figure to show the full level
    plt.imshow(level, aspect="auto", cmap="tab20")  # 20 distinct colors
    plt.colorbar(label="Tile Type")
    plt.title(title)
    plt.xlabel("Column (horizontal position)")
    plt.ylabel("Row (vertical position)")
 
    # Saves Figure
    save_path = os.path.join(OUTPUT_FIGURES, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()  # close so it doesn't display in terminal
    print(f"Saved level visualization: {save_path}")
 
 
def plot_class_distribution(Y, title="Tile Distribution", encoder=None):
    """
    Bar chart showing how many of each tile type exist in the dataset.
    """
    unique, counts = np.unique(Y, return_counts=True)
 
    plt.figure(figsize=(10, 5))
 
    # shows actual tile characters on x-axis, if we have the encoder
    if encoder:
        labels = encoder.inverse_transform(unique)
    else:
        labels = unique
 
    plt.bar(labels, counts, color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Tile Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
 
    save_path = os.path.join(OUTPUT_FIGURES, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution: {save_path}")

# TRAIN / TEST SPLIT
def split_data(X, Y):
    # Splits features and labels into training and testing sets and returns X_train, X_test, Y_train, Y_test
    # remove tile classes that appear fewer than 2 times
    unique, counts = np.unique(Y, return_counts=True)
    valid_classes = unique[counts >= 2]
    
    # keeps only samples whose label is in valid_classes
    mask = np.isin(Y, valid_classes)
    X = X[mask]
    Y = Y[mask]

    print(f"Removed {(~mask).sum()} samples with rare tile types")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test

# MODEL 1 - DECISION TREE
def train_decision_tree(X_train, Y_train):
    # Creates and trains the Decision Tree classifier and returns the trained model.

    # create the model with based on preset settings 
    dt_model = DecisionTreeClassifier(
        max_depth=10,          # doesn't let tree grow too deep
        min_samples_split=10,  # needs at least 10 samples to make a split
        random_state=RANDOM_STATE  # reproducible results
    )
 
    # trains the model so it learns from the data
    dt_model.fit(X_train, Y_train)
 
    print("Decision Tree training complete.")
    return dt_model
 
 
def visualize_tree(dt_model, encoder, max_depth_display=3):
    
   # Draws a visual of the top levels of the Decision Tree.
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        max_depth=max_depth_display,  # only shows top 3 levels
        feature_names=[f"neighbor_{i}" for i in range(9)],  # 9 neighborhood tiles
        class_names=encoder.classes_,  # shows actual tile characters
        filled=True,   # color nodes by class
        rounded=True   # prettier boxes
    )
    plt.title("Decision Tree — Top 3 Levels")
 
    save_path = os.path.join(OUTPUT_FIGURES, "decision_tree_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved tree visualization: {save_path}")

# MODEL 2: NEURAL NETWORK (MLP)
def train_neural_network(X_train, Y_train):
    """
    Creates & trains the MLP Neural Network classifier.
    Returns the trained model.
    """
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # two hidden layers: 128 then 64 neurons
        activation="relu",             # ReLU activation function
        solver="adam",                 # adaptive learning rate
        max_iter=200,                  # max training epochs
        early_stopping=True,           # stop if not improving
        validation_fraction=0.1,       # use 10% of training data to check progress
        random_state=RANDOM_STATE,
        verbose=True                   # prints progress to observe training
    )
 
    # train the model
    mlp_model.fit(X_train, Y_train)
 
    print("Neural Network training complete.")
    return mlp_model

# EVALUATE BOTH MODELS
def evaluate_model(model, X_test, Y_test, model_name, encoder):
  
   # Runs predictions & calculates all evaluation metrics, prints a full report, saves a confusion matrix figure, and returns a dict of metrics for comparison.
 
    Y_pred = model.predict(X_test)

    # only grabs the classes that actually appear in the test set
    present_classes = np.unique(np.concatenate([Y_test, Y_pred]))
    present_labels = encoder.inverse_transform(present_classes)

    accuracy  = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="weighted", zero_division=0)
    recall    = recall_score(Y_test, Y_pred, average="weighted", zero_division=0)
    f1        = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nFull Classification Report:")
    print(classification_report(
        Y_test, Y_pred,
        labels=present_classes,
        target_names=present_labels,
        zero_division=0
    ))

    plot_confusion_matrix(Y_test, Y_pred, model_name, encoder, present_classes, present_labels)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }
 
 
def plot_confusion_matrix(Y_test, Y_pred, model_name, encoder, present_classes, present_labels):
    """
    Draws & saves a heatmap confusion matrix.
    """
    cm = confusion_matrix(Y_test, Y_pred, labels=present_classes)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted Tile Type")
    plt.ylabel("Actual Tile Type")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_path = os.path.join(OUTPUT_FIGURES, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

# COMPARE & SAVE RESULTS
def save_comparison(results_list):
   # Takes a list of result dicts from evaluate_model() & saves them as a CSV comparison table.
 
    df = pd.DataFrame(results_list)  # turn list of dicts into a table
 
    print("\n" + "="*50)
    print("MODEL COMPARISON TABLE")
    print("="*50)
    print(df.to_string(index=False))  # print cleanly without row numbers
 
    # save as CSV
    save_path = os.path.join(OUTPUT_METRICS, "model_comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved comparison table: {save_path}")

# MAIN 
def main():
 
    # make sure output folders exist before trying to save to them
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    os.makedirs(OUTPUT_METRICS, exist_ok=True)
 
    # STEP 1: LOAD DATA 
    print("\n>>> STEP 1: Loading level files...")
    mario_levels   = load_levels(MARIO_PATH)
    megaman_levels = load_levels(MEGAMAN_PATH)
 
    # combine both games into one dataset
    all_levels = mario_levels + megaman_levels
    print(f"Total levels loaded: {len(all_levels)}")
 
    # STEP 2: ENCODE TILES 
    print("\n>>> STEP 2: Encoding tile characters to integers...")
    encoder, encoded_levels = encode_tiles(all_levels)
 
    # STEP 3: VISUALIZE SAMPLE LEVELS
    print("\n>>> STEP 3: Visualizing sample levels...")
    # visualize first mario level & first megaman level
    visualize_level(encoded_levels[0], "Mario Level 1: Encoded", encoder)
    visualize_level(encoded_levels[len(mario_levels)], "Mega Man Level 1: Encoded", encoder)
 
    # STEP 4: EXTRACT FEATURES
    print("\n>>> STEP 4: Extracting neighborhood features...")
    X, Y = extract_features(encoded_levels)
 
    # STEP 5: VISUALIZE CLASS DISTRIBUTION
    print("\n>>> STEP 5: Plotting tile type distribution...")
    plot_class_distribution(Y, "Tile Type Distribution: All Levels", encoder)
 
    # STEP 6: SPLIT DATA
    print("\n>>> STEP 6: Splitting into train & test sets...")
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
 
    # STEP 7: TRAIN DECISION TREE 
    print("\n>>> STEP 7: Training Decision Tree...")
    dt_model = train_decision_tree(X_train, Y_train)
    visualize_tree(dt_model, encoder)
 
    # STEP 8: TRAIN NEURAL NETWORK 
    print("\n>>> STEP 8: Training Neural Network (MLP)...")
    mlp_model = train_neural_network(X_train, Y_train)
 
    # STEP 9: EVALUATE BOTH MODELS
    print("\n>>> STEP 9: Evaluating models...")
    dt_results  = evaluate_model(dt_model,  X_test, Y_test, "Decision Tree",     encoder)
    mlp_results = evaluate_model(mlp_model, X_test, Y_test, "Neural Network MLP", encoder)
 
    # STEP 10: SAVE COMPARISON TABLE
    print("\n>>> STEP 10: Saving comparison results...")
    save_comparison([dt_results, mlp_results])
 
    print("\n>>> ALL DONE! Check outputs/ folder for all figures and metrics.")
 
# this runs main() when you execute the file directly
# WHY: prevents main() from running if this file is imported elsewhere
if __name__ == "__main__":
    main()
