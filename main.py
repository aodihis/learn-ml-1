import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

print(train.describe(include="all"))

missing_values = train.isnull().sum()
print(missing_values)
print(missing_values[missing_values > 0])

less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

categorical_features = train[less].select_dtypes(include=['object']).columns
for column in categorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

df = train.drop(columns=over)

# for feature in numeric_features:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f"Box Plot of {feature}")
#     plt.show()

q1 = df[numeric_features].quantile(0.25)
q3 = df[numeric_features].quantile(0.75)
iqr = q3 - q1

condition = ~((df[numeric_features] < (q1 - 1.5 * iqr)) | (df[numeric_features] > (q3 + 1.5 * iqr))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_features]

df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train[numeric_features[3]], kde=True)
plt.title("Histogram before Standardization")

# after
plt.subplot(1, 2, 2)
sns.histplot(train[numeric_features[3]], kde=True)
plt.title("Histogram after Standardization")

duplicates = df.duplicated()
df = df.drop_duplicates()

df_one_hot = pd.get_dummies(df, columns=categorical_features)

label_encoder = LabelEncoder()
df_lbl_encoder = pd.DataFrame(df)

for col in categorical_features:
    df_lbl_encoder[col] = label_encoder.fit_transform(df[col])

df_lbl_encoder.head()
missing_values = df_lbl_encoder.isnull().sum()
missing_percentage = missing_values / len(df_lbl_encoder) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by="Missing Values", ascending=False)


num_vars = df_lbl_encoder.shape[1]

n_cols = 4
n_rows = -(-num_vars//n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()
i = 0
for i, column in enumerate(df_lbl_encoder.columns):
    df_lbl_encoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']
plt.figure(figsize=(15,10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2,3,i)
    sns.histplot(df_lbl_encoder[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,10))
correlation_matrix = df_lbl_encoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

target_corr = df_lbl_encoder.corr()['SalePrice']

target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10,6))
target_corr_sorted.plot(kind='bar')
plt.title(f"Correlation with SalePrice")
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.show()
