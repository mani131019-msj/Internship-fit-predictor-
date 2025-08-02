
# 🧠 Internship Fit Predictor - Machine Learning Project (with 5 Models & Visualizations)

# 📦 Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# 📂 Step 2: Load the 1000-row Dataset
df = pd.read_csv("internship_fit_predictor_1000_dataset.csv")
df.columns = df.columns.str.strip()

# 🧾 Step 3: Clean and Standardize Column Names
for col in df.columns:
    if "github" in col.lower():
        df.rename(columns={col: "GitHub_Profile_Score"}, inplace=True)
    if "gpa" in col.lower():
        df.rename(columns={col: "GPA"}, inplace=True)

# 🔎 Drop unwanted object-type columns
categorical_cols = ['Skills', 'Preferred Domain', 'Certifications', 'Hackathon Participation']
non_numeric_to_drop = [
    col for col in df.select_dtypes(include='object').columns
    if col not in categorical_cols + ['Selected']
]
df = df.drop(columns=non_numeric_to_drop)

# 🏷️ Step 4: Encode the target column
df['Selected'] = df['Selected'].map({'Yes': 1, 'No': 0})

# 🔠 Step 5: Encode Categorical Columns
label_encoders = {}
for col in ['Skills', 'Preferred Domain', 'Certifications']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 🔁 Step 6: Encode Hackathon Participation
df['Hackathon Participation'] = df['Hackathon Participation'].map({'Yes': 1, 'No': 0})

# 🔢 Step 7: Normalize Numerical Columns
scaler = StandardScaler()
df[['GPA', 'GitHub_Profile_Score']] = scaler.fit_transform(df[['GPA', 'GitHub_Profile_Score']])

# 🎨 Set Seaborn Theme
sns.set_theme(style="whitegrid")

# ✅ Step 8: Enhanced Visualizations

# GPA Histogram
plt.figure(figsize=(7, 5))
sns.histplot(df['GPA'], kde=True, color='orange', edgecolor='black', linewidth=1.2)
plt.title("🎓 GPA Distribution", fontsize=14, weight='bold')
plt.xlabel("GPA", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# GitHub Score vs Selection (Boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(x='Selected', y='GitHub_Profile_Score', data=df, palette="Set2", linewidth=2)
plt.title("🐙 GitHub Score vs Selection", fontsize=14, weight='bold')
plt.xlabel("Selected (0=No, 1=Yes)", fontsize=12)
plt.ylabel("GitHub_Profile_Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Skills vs Selection Rate
plt.figure(figsize=(7, 5))
sns.barplot(x='Skills', y='Selected', data=df, palette='Spectral')
plt.title("🛠️ Skills vs Selection Rate", fontsize=14, weight='bold')
plt.xlabel("Encoded Skills", fontsize=12)
plt.ylabel("Selection Rate", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Internships Completed vs Selection
plt.figure(figsize=(7, 5))
sns.barplot(x='Internships Completed', y='Selected', data=df, palette='coolwarm')
plt.title("💼 Internships Completed vs Selection", fontsize=14, weight='bold')
plt.xlabel("Internships Completed", fontsize=12)
plt.ylabel("Selection Rate", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 🌐 Step 8.5: Network Graph from Correlation Matrix
plt.figure(figsize=(9, 7))
correlation_matrix = df.corr()
G = nx.from_pandas_adjacency(correlation_matrix)
pos = nx.spring_layout(G, seed=42)
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw(
    G, pos,
    node_color='lightgreen',
    with_labels=True,
    node_size=1600,
    font_size=10,
    width=2.5,
    edge_color=weights,
    edge_cmap=plt.cm.plasma
)
plt.title("🌐 Network Graph of Correlation Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 🧠 Step 9: Prepare Data for Modeling
X = df.drop('Selected', axis=1)
y = df['Selected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Step 10: Define and Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

# 📊 Step 11: Accuracy Comparison (Colorful Barplot)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=[results[m]['accuracy'] for m in results], palette="pastel")
plt.title("📊 Model Accuracy Comparison", fontsize=14, weight='bold')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 📉 Step 12: Heatmaps of Confusion Matrices (Colorful)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion'], annot=True, fmt='d', cmap="rocket", ax=axes[i], cbar=False)
    axes[i].set_title(name, fontsize=10, weight='bold')
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.suptitle("📉 Confusion Matrices for All Models", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

results[name] = {
    "model": model,
    "accuracy": acc,
    "confusion": confusion_matrix(y_test, y_pred),
    "report": classification_report(y_test, y_pred, output_dict=True)
}

sns.barplot(x=list(results.keys()), y=[results[m]['accuracy'] for m in results], palette="pastel")
# 🔍 Print all accuracy scores
print("\n🔢 Accuracy Scores of All Models:")
for name, res in results.items():
    print(f"{name}: {res['accuracy']:.4f}")

# ✅ Identify the best performing model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Best Performing Model: {best_model[0]} with Accuracy = {best_model[1]['accuracy']:.4f}")

# ✅ Step 13: Test on New Student Data (Robust Version)
def predict_fit(student_data, model_name="KNN"):
    sample_df = pd.DataFrame([student_data])
    sample_df.columns = sample_df.columns.str.strip()

    # Safely encode categorical columns using trained LabelEncoders
    for col in ['Skills', 'Preferred Domain', 'Certifications']:
        le = label_encoders[col]
        original_value = sample_df[col][0]
        if original_value not in le.classes_:
            print(f"⚠️ Warning: Unseen value '{original_value}' in column '{col}'. Assigning as 'Other'")
            if "Other" in le.classes_:
                sample_df[col] = le.transform(["Other"])
            else:
                # Append 'Other' to encoder classes if not already present
                le.classes_ = np.append(le.classes_, 'Other')
                sample_df[col] = le.transform(["Other"])
        else:
            sample_df[col] = le.transform([original_value])

    # Encode Hackathon Participation
    sample_df['Hackathon Participation'] = 1 if sample_df['Hackathon Participation'][0] == 'Yes' else 0

    # Normalize numerical columns
    sample_df[['GPA', 'GitHub_Profile_Score']] = scaler.transform(sample_df[['GPA', 'GitHub_Profile_Score']])

    # Ensure correct feature order
    sample_df = sample_df[X.columns.tolist()]

    # Predict using chosen model
    model = results[model_name]['model']
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample_df)[0][1]
    else:
        prob = model.predict(sample_df)[0]  # fallback if no probability support

    print(f"\n🔍 Using Model: {model_name}")
    print(f"🔮 Probability of Selection: {prob:.2f}" if hasattr(model, "predict_proba") else f"🔮 Prediction: {prob}")
    if prob >= 0.5:
        print("✅ The student IS a good fit for the internship.")
    else:
        print("❌ The student is NOT a fit for the internship.")

# 🚀 Step 14: Run Test Case

print("\n--- Test Case 1 ---")
new_student_1 = {
    'GPA': 8.5,
    'Skills': 'Python, ML',
    'Projects': 4,
    'Preferred Domain': 'Data Science',
    'Certifications': 'Coursera ML',
    'Internships Completed': 5,
    'Hackathon Participation': 'No',
    'GitHub_Profile_Score': 8
}
predict_fit(new_student_1)

# ✅ Step 15: Completion
print("\n✅ Project Completed: All models trained and evaluated with visualizations.")