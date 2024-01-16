# Download Iris dataset from the MIT OpenCourseWare
# URL:https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/resources/iris/
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Set the matplotlib backend to "TkAgg" to ensure proper display of plots in PyCharm.
matplotlib.use("TkAgg")

# Read the iris.csv dataset and prevent treating the first row as header
df = pd.read_csv('Iris/iris.csv', header=None)

# Define the column names
column_names = [
    "sepalLength",
    "sepalWidth",
    "petalLength",
    "petalWidth",
    "species"
]

# Assign the column names to the DataFrame
df.columns = column_names

# Define the mapping
species_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

# Map the encoded values to the desired ones
df['species'] = df['species'].map(species_mapping)

# Get all rows of the DataFrame
print(df.to_string())
print(df.info())

# Check if there are any missing values in any column
print(df.isnull().any())
# Print information about the DataFrame
print(df.isnull().sum())

# Create a boolean Series indicating whether each row is a duplicate or not
duplicates = df.duplicated()

# Use boolean indexing to get all occurrences of duplicated rows
duplicated_rows = df[df.duplicated(keep=False)]

# Print all duplicated rows
print(duplicated_rows)

# Remove duplicated rows
df.drop_duplicates(inplace=True)

# Print information about the DataFrame to confirm the status change
print(df.info())


min_data=df["sepalLength"].min()
max_data=df["sepalLength"].max()
sum_data = df["sepalLength"].sum()
mean_data = df["sepalLength"].mean
median_data = df["sepalLength"].median

print("Minimum:",min_data, "\nMaximum:", max_data,"Sum:", sum_data, "\nMean:", mean_data, "\nMedian:", median_data)

# Use the corr() method and exclude non-numeric columns
# print(df.drop('species', axis=1).corr())
print(df.corr())

# Plot the entire DataFrame
df.plot()
plt.show()

# Create a scatter plot with 'sepalLength' on the x-axis, 'petalLength' on the y-axis
df.plot(kind = 'scatter', x = 'sepalLength', y = 'petalLength')
plt.show()

sns.heatmap(df.drop('species', axis=1).corr(), cmap="YlGnBu")
plt.show()


# 'X' contains features and 'y' contains target variable 'species'
X = df.drop('species', axis=1)
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Initialize kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn_classifier.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix for Test Set
cm = confusion_matrix(y_test, y_pred)
# Create a heatmap for the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# Create a heatmap for the confusion matrix
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
# Convert confusion matrix to a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=species_mapping.keys(), columns=species_mapping.keys())
print(cm_df)

# Initialize kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Perform 10-fold cross-validation
cross_val_scores = cross_val_score(knn_classifier, X, y, cv=10)

# Print the accuracy for each fold
print("Cross-Validation Scores:", cross_val_scores)

# Print the mean accuracy across all folds
print("Mean Accuracy:", cross_val_scores.mean())

# Create a boxplot to visualize the distribution of cross-validation accuracy scores
sns.boxplot(x=cross_val_scores)
# Set labels and title for better visualization
plt.xlabel("Cross-Validation Folds")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy Distribution")
plt.show()