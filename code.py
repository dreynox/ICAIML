# ----- Cell 1 -----
# ============================================================
# Environment Setup & Reproducibility
# ============================================================
# This notebook implements a research-grade machine learning
# framework for plant health prediction in agro-ecosystems.
#
# Key objectives:
# - Comparative evaluation of multiple ML models
# - Cross-validation for reliable performance estimation
# - Robustness analysis under noisy and incomplete data
# - Explainable AI for agronomic interpretability
# ============================================================
# Fix random seed for reproducibility
# Note:
# Dataset files are expected to be available locally
# or mounted from a cloud storage bucket.

# ----- Cell 4 -----
# ============================================================
# Core Libraries
# ============================================================
import numpy as np
import pandas as pd
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# ============================================================
# Visualization
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Statistical & Preprocessing Utilities
# ============================================================
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# ============================================================
# Model Selection & Validation
# ============================================================
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

# ============================================================
# Evaluation Metrics
# ============================================================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from collections import Counter

# ============================================================
# Machine Learning Models
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC

# ============================================================
# Explainable AI
# ============================================================
import shap

# ============================================================
# Robustness & Experiment Control
# ============================================================
import warnings
warnings.filterwarnings("ignore")

# ----- Cell 6 -----
# Load the dataset
df = pd.read_csv('plant_health_data.csv')
# Define target
y = df["Plant_Health_Status"]

# Define features
X = df.drop(columns=["Plant_Health_Status"])

# 🔴 PASTE THIS HERE (diagnostic test)
suspect_features = [
    "Electrochemical_Signal",
    "Chlorophyll_Content",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level"
]

X = X.drop(columns=suspect_features)

# ----- Cell 7 -----
# Display basic information about the dataset
print("Shape of the dataset:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().T)

# ----- Cell 9 -----
# Check for missing and duplicated values
print("Total missing values:", df.isna().sum().sum())
print("Total duplicated rows:", df.duplicated().sum())

# ----- Cell 11 -----
# Display the number of unique values in each column
print("Unique values per column:")
print(df.nunique())

# ----- Cell 12 -----
# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Display feature groups
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

# ----- Cell 13 -----
# Display unique values for each categorical column
for col in categorical_columns:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].unique())

# ----- Cell 16 -----
column_name = 'Plant_Health_Status'

# Ensure consistent class order
class_order = df[column_name].value_counts().index

plt.figure(figsize=(10, 4))

# --------------------------------------------------
# Count plot
# --------------------------------------------------
plt.subplot(1, 2, 1)
sns.countplot(
    y=column_name,
    data=df,
    order=class_order
)
plt.title(f'Distribution of {column_name}')

ax = plt.gca()
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_width())}',
        (p.get_width(), p.get_y() + p.get_height() / 2),
        ha='left',
        va='center',
        xytext=(5, 0),
        textcoords='offset points'
    )

sns.despine(left=True, bottom=True)

# --------------------------------------------------
# Pie chart
# --------------------------------------------------
plt.subplot(1, 2, 2)
df[column_name].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    explode=[0.05] * df[column_name].nunique()
)
plt.title(f'Percentage Distribution of {column_name}')
plt.ylabel('')

plt.tight_layout()
plt.show()

# ----- Cell 18 -----
# Find the earliest and latest timestamps
start_date = pd.to_datetime(df['Timestamp']).min()
end_date = pd.to_datetime(df['Timestamp']).max()

print("Start Date:", start_date)
print("End Date:", end_date)

# ----- Cell 19 -----
# Convert Timestamp to datetime for exploratory analysis
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Analyze time intervals between consecutive measurements
time_diffs = df['Timestamp'].diff().value_counts()

print("Most common time differences between measurements:")
print(time_diffs.head(10))

# ----- Cell 21 -----
# Function to perform univariate analysis for numeric sensor-based columns
def univariate_analysis(data, columns):
    plt.figure(figsize=(16, 24))

    muted_colors = sns.color_palette("muted", len(columns))

    for i, column in enumerate(columns):
        plt.subplot(4, 3, i + 1)
        sns.histplot(data[column], kde=True, bins=10, color=muted_colors[i])
        plt.title(f'{column.replace("_", " ")} Distribution with KDE')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Sensor-based numerical features only (excluding identifiers)
columns_to_analyze = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal'
]

# Perform univariate analysis
univariate_analysis(df, columns_to_analyze)

# ----- Cell 23 -----
# Select numerical sensor-based columns only (exclude identifiers)
numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['Plant_ID'])

# Calculate skewness and kurtosis
skewness = numerical_df.skew()
kurtosis = numerical_df.kurt()

display("Skewness:", skewness)
print("\n")
display("Kurtosis:", kurtosis)

# ----- Cell 25 -----
# Function to perform univariate analysis for numeric sensor-based columns
def univariate_analysis(data, column, title):
    plt.figure(figsize=(10, 2))

    color = sns.color_palette("muted")[columns_to_analyze.index(column) % len(sns.color_palette("muted"))]

    sns.boxplot(x=data[column], color=color)
    plt.title(f'{title} Boxplot')

    plt.tight_layout()
    plt.show()

    print(f'\nSummary Statistics for {title}:\n', data[column].describe())

# Sensor-based numerical features only (excluding identifiers)
columns_to_analyze = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal'
]

# Iterate through columns and perform univariate analysis
for column in columns_to_analyze:
    univariate_analysis(df, column, column.replace('_', ' '))

# ----- Cell 28 -----
# Convert Timestamp to datetime format for temporal aggregation
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create a weekly grouping variable (used only for exploratory analysis)
df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)

# Aggregate Plant Health Status weekly for each Plant ID (exploratory analysis)
weekly_health_status = (
    df.groupby(['Plant_ID', 'Week', 'Plant_Health_Status'])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

weekly_health_status.columns.name = None
weekly_health_status = weekly_health_status.rename(columns={
    'High Stress': 'High_Stress_Count',
    'Moderate Stress': 'Moderate_Stress_Count',
    'Healthy': 'Healthy_Count'
})

weekly_health_status

# ----- Cell 29 -----
# Compare Plant_Health_Status with Soil Properties

# Define soil properties
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH',
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

# Consistent class order
class_order = ['High Stress', 'Moderate Stress', 'Healthy']

# Create subplots to visualize the relationship between Plant_Health_Status and soil properties
plt.figure(figsize=(16, 20))

for i, feature in enumerate(soil_properties):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(
        x='Plant_Health_Status',
        y=feature,
        data=df,
        order=class_order,
        palette='muted'
    )
    plt.title(f'{feature.replace("_", " ")} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ----- Cell 31 -----
# Compare Plant_Health_Status with Environmental Conditions

# Define environmental condition features
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

# Consistent class order
class_order = ['High Stress', 'Moderate Stress', 'Healthy']

# Create subplots to visualize the relationship between Plant_Health_Status and environmental conditions
plt.figure(figsize=(16, 12))

for i, feature in enumerate(environmental_conditions):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(
        x='Plant_Health_Status',
        y=feature,
        data=df,
        order=class_order,
        palette='muted'
    )
    plt.title(f'{feature.replace("_", " ")} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ----- Cell 33 -----
# Compare Plant_Health_Status with Plant Health Indicators

# Define plant health indicator features
health_indicators = [
    'Chlorophyll_Content',
    'Electrochemical_Signal'
]

# Fix class order for consistency
class_order = ['Healthy', 'Moderate Stress', 'High Stress']

# Define consistent color palette
palette = sns.color_palette("muted", len(class_order))

# Create subplots
plt.figure(figsize=(16, 8))
for i, feature in enumerate(health_indicators):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(
        x='Plant_Health_Status',
        y=feature,
        data=df,
        order=class_order,
        palette=palette
    )
    plt.title(f'{feature.replace("_", " ")} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ----- Cell 35 -----
# Compare Plant_Health_Status with Plant Health Indicators vs Soil Properties using scatter plots

# Define soil properties and plant health indicators
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH',
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Consistent color mapping for health status
status_palette = {
    'Healthy': '#2ecc71',
    'Moderate Stress': '#f1c40f',
    'High Stress': '#e74c3c'
}

# Create scatter plots
plt.figure(figsize=(30, 15))
plot_index = 1

for health_indicator in health_indicators:
    for soil_property in soil_properties:
        ax = plt.subplot(len(health_indicators), len(soil_properties), plot_index)

        sns.scatterplot(
            x=soil_property,
            y=health_indicator,
            hue='Plant_Health_Status',
            data=df,
            palette=status_palette,
            alpha=0.7,
            ax=ax,
            legend=(plot_index == 1)  # show legend only once
        )

        plt.title(f'{health_indicator} vs {soil_property}')
        plt.xlabel(soil_property.replace('_', ' '))
        plt.ylabel(health_indicator.replace('_', ' '))

        plot_index += 1

plt.tight_layout()
plt.show()

# ----- Cell 37 -----
# Compare Plant_Health_Status with Plant Health Indicators vs Environmental Conditions using scatter plots

# Define environmental conditions and plant health indicators
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Create scatter plots for each combination of plant health indicators and environmental conditions
plt.figure(figsize=(20, 12))
plot_index = 1

for health_indicator in health_indicators:
    for env_condition in environmental_conditions:
        plt.subplot(len(health_indicators), len(environmental_conditions), plot_index)
        sns.scatterplot(
            x=env_condition,
            y=health_indicator,
            hue='Plant_Health_Status',
            data=df,
            palette='muted',
            alpha=0.7
        )
        plt.title(f'{health_indicator} vs {env_condition}')
        plt.xlabel(env_condition.replace('_', ' '))
        plt.ylabel(health_indicator.replace('_', ' '))
        plt.legend(title='Health Status', loc='upper right')
        plot_index += 1

plt.tight_layout()
plt.show()


# ----- Cell 38 -----
# Statistical profiling of plant health indicators and environmental conditions
# across different plant health statuses

analysis_features = [
    'Chlorophyll_Content',
    'Electrochemical_Signal',
    'Ambient_Temperature',
    'Humidity',
    'Light_Intensity'
]

# Compute descriptive statistics
health_status_profile = (
    df
    .groupby('Plant_Health_Status')[analysis_features]
    .agg(['mean', 'std', 'median'])
    .round(2)
)

# Rename aggregation levels for clarity
health_status_profile.columns = [
    f"{feature}_{stat.upper()}"
    for feature, stat in health_status_profile.columns
]

print("=== Descriptive Statistics by Plant Health Status ===")
display(health_status_profile)

# ----- Cell 40 -----
# ==========================================
# Statistical analysis of soil properties
# across plant health categories
# ==========================================

soil_features = [
    'Soil_Moisture',
    'Soil_Temperature',
    'Soil_pH',
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

# Compute descriptive statistics
soil_status_profile = (
    df
    .groupby('Plant_Health_Status')[soil_features]
    .agg(['mean', 'std', 'median'])
    .round(2)
)

# Flatten column names for readability and reporting
soil_status_profile.columns = [
    f"{feature}_{stat.upper()}"
    for feature, stat in soil_status_profile.columns
]

print("=== Descriptive Statistics of Soil Properties by Plant Health Status ===")
display(soil_status_profile)

# ----- Cell 42 -----
# ==========================================
# Plant-wise analysis of soil nutrient availability
# ==========================================

nutrient_features = [
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

# Compute descriptive statistics of nutrients for each plant
plant_nutrient_profile = (
    df
    .groupby('Plant_ID')[nutrient_features]
    .agg(['mean', 'std', 'median'])
    .round(2)
)

# Flatten column names for clarity
plant_nutrient_profile.columns = [
    f"{feature}_{stat.upper()}"
    for feature, stat in plant_nutrient_profile.columns
]

print("=== Plant-wise Nutrient Statistics ===")
display(plant_nutrient_profile)

# ----- Cell 44 -----
# Distribution of soil nutrient levels across individual plants

nutrient_features = [
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(nutrient_features),
    figsize=(18, 6),
    sharey=False
)

for ax, nutrient in zip(axes, nutrient_features):
    sns.boxplot(
        data=df,
        x='Plant_ID',
        y=nutrient,
        palette='muted',
        ax=ax
    )

    ax.set_title(f'{nutrient.replace("_", " ")} Distribution')
    ax.set_xlabel('Plant ID')
    ax.set_ylabel(nutrient.replace('_', ' '))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ----- Cell 46 -----
# Mean nutrient levels by Plant ID and Plant Health Status

nutrient_features = [
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

nutrient_status_mean = (
    df
    .groupby(['Plant_ID', 'Plant_Health_Status'])[nutrient_features]
    .mean()
    .round(2)
    .reset_index()
)

# Separate nutrient profiles by Plant Health Status

nutrient_profiles = {
    'High Stress': nutrient_status_mean.query("Plant_Health_Status == 'High Stress'"),
    'Moderate Stress': nutrient_status_mean.query("Plant_Health_Status == 'Moderate Stress'"),
    'Healthy': nutrient_status_mean.query("Plant_Health_Status == 'Healthy'")
}

for status, table in nutrient_profiles.items():
    print(f"\n----- Nutrient Levels for {status} Plants -----")
    display(table)

# ----- Cell 48 -----
# Visualize the nutrient levels across different Plant_Health_Status categories (High Stress, Moderate Stress, Healthy) for each Plant_ID
# Define nutrient levels
nutrients = ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']

# Set up the plot
plt.figure(figsize=(18, 12))
for i, nutrient in enumerate(nutrients):
    plt.subplot(2, 2, i + 1)
    sns.barplot(
        x='Plant_ID',
        y=nutrient,
        hue='Plant_Health_Status',
        data=nutrient_status_mean,
        palette='muted'
    )
    plt.title(f'{nutrient.replace("_", " ")} by Plant ID and Health Status')
    plt.xlabel('Plant ID')
    plt.ylabel(nutrient.replace('_', ' '))
    plt.legend(title='Health Status', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# ----- Cell 49 -----
# Analysis of Environmental Conditions vs Plant Health Status
# Using normalized distributions for better cross-class comparison

environmental_conditions = [
    'Ambient_Temperature',
    'Humidity',
    'Light_Intensity'
]

for condition in environmental_conditions:
    sns.displot(
        data=df,
        x=condition,
        col='Plant_Health_Status',
        kind='hist',
        kde=True,
        bins=12,
        stat='density',
        height=4,
        aspect=1.1,
        palette='muted'
    )

    plt.suptitle(
        f'Distribution of {condition.replace("_", " ")} Across Plant Health Categories',
        fontsize=15,
        y=1.05
    )

    plt.show()

# ----- Cell 51 -----
# Distribution analysis of plant health indicators across health categories

health_indicators = ['Chlorophyll_Content', 'Electrochemical_Signal']

for indicator in health_indicators:

    # Initialize FacetGrid
    grid = sns.FacetGrid(
        data=df,
        col='Plant_Health_Status',
        col_order=['Healthy', 'Moderate Stress', 'High Stress'],
        height=4.2,
        aspect=1.1,
        despine=False
    )

    # Plot histogram with KDE
    grid.map_dataframe(
        sns.histplot,
        x=indicator,
        bins=12,
        kde=True,
        stat='density',
        alpha=0.75
    )

    # Axis labels and titles
    grid.set_axis_labels(
        indicator.replace('_', ' '),
        'Density'
    )
    grid.set_titles(col_template='{col_name}')

    # Global title
    grid.fig.suptitle(
        f'Health-wise Distribution of {indicator.replace("_", " ")}',
        fontsize=16,
        fontweight='bold'
    )

    grid.fig.subplots_adjust(top=0.82)
    plt.show()

# ----- Cell 53 -----
# Distribution of Soil Properties across Plant Health Categories
# FacetGrid-based visualization for comparative soil analysis

soil_features = [
    'Soil_Moisture',
    'Soil_Temperature',
    'Soil_pH',
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

for feature in soil_features:
    grid = sns.FacetGrid(
        data=df,
        col='Plant_Health_Status',
        col_order=['Healthy', 'Moderate Stress', 'High Stress'],
        height=3.8,
        aspect=1.3,
        sharex=False,
        sharey=False
    )

    grid.map_dataframe(
        sns.histplot,
        x=feature,
        kde=True,
        bins=12,
        alpha=0.75
    )

    grid.set_axis_labels(
        feature.replace('_', ' '),
        'Sample Count'
    )

    grid.set_titles(
        template='{col_name}'
    )

    grid.fig.suptitle(
        f'Soil Property Distribution by Plant Health Status: {feature.replace("_", " ")}',
        fontsize=15,
        fontweight='bold',
        y=1.05
    )

    plt.show()

# ----- Cell 55 -----
# ================================
# Correlation Analysis of Numerical Features
# ================================

# Select numerical features only
numeric_features = df.select_dtypes(include=np.number)

# Compute Pearson correlation matrix
corr_matrix = numeric_features.corr(method='pearson')

# Create the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.6,
    linecolor='white',
    cbar_kws={
        "label": "Correlation Coefficient",
        "shrink": 0.85
    }
)

# Plot formatting
plt.title(
    "Correlation Structure Among Numerical Agro-Physiological Features",
    fontsize=16,
    pad=15
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# ----- Cell 57 -----
# ================================
# Time-Series Preparation & Aggregation
# ================================

# Ensure timestamp column is in datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort data chronologically before indexing
df = df.sort_values('Timestamp')

# Set Timestamp as the time index
df = df.set_index('Timestamp')

# Select numerical variables relevant for temporal analysis
temporal_features = [
    'Soil_Moisture',
    'Soil_Temperature',
    'Ambient_Temperature',
    'Humidity',
    'Soil_pH',
    'Light_Intensity',
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level'
]

# Compute aggregated statistics at different temporal resolutions
daily_avg_features = (
    df[temporal_features]
    .resample('D')
    .mean()
)

weekly_avg_features = (
    df[temporal_features]
    .resample('W')
    .mean()
)

# Preview results
display(daily_avg_features.head())
display(weekly_avg_features.head())

# ----- Cell 58 -----
# ================================
# Daily Trend Visualization of Key Environmental Features
# ================================

# Select a subset of features for clearer visualization
selected_features = temporal_features[:5]

# Initialize the figure
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(16, 12),
    sharex=True
)

axes = axes.flatten()

# Plot daily trends for each selected feature
for idx, feature in enumerate(selected_features):
    daily_avg_features[feature].plot(
        ax=axes[idx],
        linewidth=2
    )
    axes[idx].set_title(f'Daily Variation in {feature.replace("_", " ")}')
    axes[idx].set_ylabel(feature.replace('_', ' '))
    axes[idx].grid(alpha=0.3)

# Remove unused subplot (since we plot only 5 features)
fig.delaxes(axes[-1])

# Improve layout
plt.suptitle(
    'Daily Temporal Trends of Environmental and Soil Features',
    fontsize=16,
    y=1.02
)
plt.tight_layout()
plt.show()

# ----- Cell 60 -----
# Weekly trend analysis for soil chemistry and nutrient-related features

# Select remaining features for weekly analysis
weekly_features = temporal_features[5:]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Plot weekly averages
for ax, feature in zip(axes, weekly_features):
    weekly_avg_features[feature].plot(ax=ax)
    ax.set_title(f'Weekly Trend of {feature.replace("_", " ")}', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel(feature.replace('_', ' '))
    ax.grid(True, linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout()
plt.show()

# ----- Cell 63 -----
# ==========================================
# Encoding Plant Health Status for ML Models
# ==========================================

# Define ordinal mapping (target variable encoding)
custom_mapping = {
    'High Stress': 2,
    'Moderate Stress': 1,
    'Healthy': 0
}

# Apply encoding
df['Plant_Health_Status_Encoded'] = df['Plant_Health_Status'].map(custom_mapping)

# Safety check: ensure no unmapped values
if df['Plant_Health_Status_Encoded'].isnull().sum() > 0:
    print("Warning: Some values were not mapped correctly!")

# Value counts BEFORE encoding
print("----- Class Distribution (Original Labels) -----")
print(df['Plant_Health_Status'].value_counts())

# Value counts AFTER encoding
print("\n----- Class Distribution (Encoded Labels) -----")
print(df['Plant_Health_Status_Encoded'].value_counts().sort_index())

# ----- Cell 64 -----
# Select only meaningful numerical features (exclude ID + target if needed)
numerical_features = df.select_dtypes(include=[np.number]).columns

# Remove target column if present
numerical_features = numerical_features.drop(
    ['Plant_Health_Status_Encoded'],
    errors='ignore'
)

# Compute Z-scores
z_scores = zscore(df[numerical_features])

# Count outliers (|z| > 3)
outliers_zscore = (np.abs(z_scores) > 3).sum(axis=0)

# Display results
print("===== Outliers Detected Using Z-Score Method (|z| > 3) =====")
print(outliers_zscore)

# ----- Cell 66 -----
# Dropping non-numerical / redundant columns

# Drop original categorical target and temporal feature
df = df.drop(columns=['Plant_Health_Status', 'Week'], errors='ignore')

print("Remaining columns after drop:")
print(df.columns)

# ----- Cell 67 -----
# Correlation of features with target variable

numerical_features = df.select_dtypes(include=[np.number])

correlations = numerical_features.corr()['Plant_Health_Status_Encoded'].sort_values(ascending=False)

# Convert to DataFrame for better readability
correlation_table = correlations.to_frame(
    name='Correlation with Plant_Health_Status_Encoded'
).reset_index()

correlation_table.rename(columns={'index': 'Feature'}, inplace=True)

display(correlation_table)

# ----- Cell 68 -----
# Feature interaction with encoded plant health status

# Ensure only numeric features are used (excluding target leakage issues)
features_only = df.select_dtypes(include=[np.number]).drop(
    columns=['Plant_Health_Status_Encoded'], errors='ignore'
)

status_correlation = df.groupby('Plant_Health_Status_Encoded')[features_only.columns].mean().T

plt.figure(figsize=(10, 8))

sns.heatmap(
    status_correlation,
    annot=True,
    cmap='YlGnBu_r',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.9}
)

plt.title('Feature Interaction with Plant Health Status (Encoded)')
plt.ylabel('Features')
plt.xlabel('Plant Health Status Encoded')

plt.tight_layout()
plt.show()

# ----- Cell 70 -----
# Define features and target variable

# Features (independent variables)
X = df.drop(columns=['Plant_Health_Status_Encoded'])

# Target (dependent variable)
y = df['Plant_Health_Status_Encoded']

# ----- Cell 71 -----
# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Verification
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Class distribution check (important)
print("\nTrain class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# ----- Cell 72 -----
# Create a scaler pipeline
scaling_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Fit ONLY on training data
scaling_pipeline.fit(X_train)

# Transform train and test data
X_train_scaled = scaling_pipeline.transform(X_train)
X_test_scaled = scaling_pipeline.transform(X_test)

# ----- Cell 74 -----
# Initialize models using pipelines (scaling only where required)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "KNN Classifier": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "Support Vector Machine (SVM)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42
        ))
    ]),

    "Extra Trees": Pipeline([
        ("model", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42
        ))
    ])
}

# Display model names to confirm initialization
print("Models initialized:", list(models.keys()))

# ----- Cell 75 -----
# Initialize list to store results
results = []

# Class names (must match label encoding order)
class_names = ['Healthy', 'Moderate Stress', 'High Stress']

# Train and evaluate each model (PIPELINE-SAFE)
for model_name, model in models.items():

    # Train model (pipeline handles scaling internally if needed)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # AUC (proper multi-class handling)
    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test, y_pred, target_names=class_names
        )
    })

# ----------------------------
# Display Results
# ----------------------------
for result in results:
    print(f"\nModel: {result['Model']}")
    print(f"Accuracy: {result['Accuracy']:.3f}")

    if result["AUC"] is not None:
        print(f"AUC (OvR): {result['AUC']:.3f}")

    # Confusion Matrix
    cm = result["Confusion Matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {result['Model']}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Classification Report
    print(result["Classification Report"])

# ----- Cell 77 -----
# Prepare a summary table for model evaluation
evaluation_summary = []

# Extract the key metrics for each model
for result in results:
    evaluation_summary.append({
        "Model": result["Model"],
        "Accuracy": result["Accuracy"],
        "AUC": result["AUC"]
    })

evaluation_summary_df = pd.DataFrame(evaluation_summary)

# Sort by accuracy and display all models used
evaluation_summary_df = evaluation_summary_df.sort_values(by="Accuracy", ascending=False)

display(evaluation_summary_df)

# ----- Cell 78 -----
# Select the Model with the Highest AUC (safe handling for None values)
best_result = max(
    results,
    key=lambda x: x["AUC"] if x["AUC"] is not None else -1
)

best_model_name = best_result["Model"]
best_model = models[best_model_name]

print(f"Best model based on AUC: {best_model_name}")

# ----- Cell 79 -----
# Plot Feature Importances (for tree-based models only)
if hasattr(best_model, "feature_importances_"):

    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))

    colors = plt.cm.YlGnBu_r(np.linspace(0, 1, len(feature_importances)))

    plt.barh(
        range(len(sorted_idx)),
        feature_importances[sorted_idx],
        align='center',
        color=colors
    )

    plt.yticks(
        range(len(sorted_idx)),
        [feature_names[i] for i in sorted_idx]
    )

    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importances - {best_model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

else:
    print(f"{best_model_name} does not support built-in feature importance.")

# ----- Cell 81 -----
# ============================================================
# Generate Predictions using Best Model
# ============================================================

y_pred_best = best_model.predict(X_test_scaled)

# ============================================================
# Visualize Prediction Distribution
# ============================================================

plt.figure(figsize=(8, 6))

ax = sns.countplot(
    x=y_pred_best,
    palette='YlGnBu',
    order=[0, 1, 2]
)

# Replace numeric labels with class names
ax.set_xticklabels(["Healthy", "Moderate Stress", "High Stress"])

plt.xlabel("Predicted Plant Health Status")
plt.ylabel("Number of Samples")
plt.title(f"Prediction Distribution - {best_model_name}")

# Add value labels on top of bars (better readability)
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# ----- Cell 82 -----
# ============================================================
# Actual vs Predicted Distribution Comparison
# ============================================================

plt.figure(figsize=(10, 6))

# Create comparison DataFrame
actual_vs_predicted = pd.DataFrame({
    'Actual': y_test,
    'Predicted': best_model.predict(X_test_scaled)
})

# Convert to long format for grouped visualization
plot_data = actual_vs_predicted.melt(
    var_name='Type',
    value_name='Plant Health Status'
)

ax = sns.countplot(
    data=plot_data,
    x='Plant Health Status',
    hue='Type',
    palette='YlGnBu',
    order=[0, 1, 2]
)

# Replace numeric labels with class names
ax.set_xticklabels(["Healthy", "Moderate Stress", "High Stress"])

plt.xlabel("Plant Health Status")
plt.ylabel("Number of Samples")
plt.title("Actual vs Predicted Plant Health Status Distribution")

plt.legend(title="Type", loc="upper left")

# Add value labels for better interpretability
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# ----- Cell 84 -----
# Cross-Validation (Robust Performance Check)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for model_name, model in models.items():
    scores = cross_validate(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr'],
        return_train_score=False
    )

    cv_results.append({
        "Model": model_name,
        "CV Accuracy Mean": np.mean(scores['test_accuracy']),
        "CV F1 Mean": np.mean(scores['test_f1_weighted']),
        "CV AUC Mean": np.mean(scores['test_roc_auc_ovr'])
    })

cv_results

# ----- Cell 85 -----
# Display CV Results Cleanly
cv_df = pd.DataFrame(cv_results)
cv_df.sort_values(by="CV Accuracy Mean", ascending=False)

# ----- Cell 86 -----
# Add Gaussian noise to test robustness
noise_factor = 0.05  # 5% noise

X_test_noisy = X_test_scaled + np.random.normal(
    loc=0,
    scale=noise_factor,
    size=X_test_scaled.shape
)

noise_results = []

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred_clean = model.predict(X_test_scaled)
    y_pred_noisy = model.predict(X_test_noisy)

    acc_clean = accuracy_score(y_test, y_pred_clean)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)

    noise_results.append({
        "Model": model_name,
        "Accuracy (Clean)": acc_clean,
        "Accuracy (Noisy)": acc_noisy,
        "Drop": acc_clean - acc_noisy
    })

pd.DataFrame(noise_results).sort_values(by="Drop", ascending=False)

# ----- Cell 87 -----
# Select final model (after CV + robustness checks)
# Select best model (Decision Tree)
model = models["Decision Tree"]

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# ============================================================
# Global Explanation (Feature Importance View)
# ============================================================

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X.columns
)

# ============================================================
# Local Explanation (Single Prediction - SAFE VERSION)
# ============================================================

sample_idx = 0

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0][sample_idx],  # class 0 example
        base_values=explainer.expected_value[0],
        data=X_test_scaled[sample_idx],
        feature_names=X.columns
    )
)

