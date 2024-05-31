import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import f_oneway
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('processed_data/100k_population_data.csv', low_memory=False)

# Exclude territories
territories_to_exclude = ['US_GU', 'US_VI', 'US_AS', 'US_PR', 'US_MP', 'US_DC']
df = df[~df['location_key'].isin(territories_to_exclude)]

# Define state lists
blue_states_list = ['US_CA', 'US_ME', 'US_OR', 'US_CO', 'US_MD', 'US_RI', 'US_CT', 'US_MA', 'US_VT', 'US_DE', 'US_NH', 'US_VA', 'US_NJ', 'US_WA', 'US_HI', 'US_NM', 'US_NY', 'US_IL']
red_states_list = ['US_AL', 'US_AK', 'US_AR', 'US_ID', 'US_IN', 'US_IA', 'US_KS', 'US_KY', 'US_LA', 'US_MS', 'US_MO', 'US_MT', 'US_NE', 'US_ND', 'US_OK', 'US_SC', 'US_SD', 'US_TN', 'US_TX', 'US_UT', 'US_WV', 'US_WY']
swing_states_list = ['US_AZ', 'US_NV', 'US_FL', 'US_NC', 'US_GA', 'US_OH', 'US_MI', 'US_PA', 'US_MN', 'US_WI']

# Create dummy variables for state types
df['blue_states'] = df['location_key'].apply(lambda x: 1 if x in blue_states_list else 0)
df['red_states'] = df['location_key'].apply(lambda x: 1 if x in red_states_list else 0)
df['swing_states'] = df['location_key'].apply(lambda x: 1 if x in swing_states_list else 0)

# Create a state_type column
def get_state_type(row):
    if row['blue_states'] == 1:
        return 'Blue'
    elif row['red_states'] == 1:
        return 'Red'
    elif row['swing_states'] == 1:
        return 'Swing'
    else:
        return 'Other'

df['state_type'] = df.apply(get_state_type, axis=1)

# Drop unneeded columns
columns_to_drop = ['blue_states', 'red_states', 'swing_states', 'target_end_date', 'location_key', 'location', 'new_hospitalized_patients', 'hospitalized_per_100k', 'unemployment_rate', 'year']

# Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Dropping the nan in specific columns
df = df.dropna(subset=['new_persons_fully_vaccinated', 'vaccinated_per_100k'])

# Display dataframe
display(df)

# Split data into features (X) and target variable (y)
X = df.drop(columns=['cases_per_100k', 'total_population', 'inc cases'])
y = df['cases_per_100k'] 

# Define smoter categorical function
def smoter_categorical(X, y, state_type, minority_class, k_neighbors=5, new_samples=100):
    # Identify minority samples based on the categorical condition
    minority_indices = np.where(state_type == minority_class)[0]
    minority_samples = X[minority_indices]
    minority_targets = y[minority_indices]
    
    # Fit nearest neighbors on minority samples
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(minority_samples)
    synthetic_samples = []
    synthetic_targets = []
    
    for _ in range(new_samples):
        # Randomly choose a minority sample
        idx = np.random.choice(minority_indices)
        sample = minority_samples[np.where(minority_indices == idx)[0][0]]
        target = minority_targets[np.where(minority_indices == idx)[0][0]]
        
        # Find neighbors and create a synthetic sample
        _, neighbors = nbrs.kneighbors([sample])
        neighbor_idx = neighbors[0][np.random.randint(1, k_neighbors)]
        neighbor_sample = minority_samples[neighbor_idx]
        neighbor_target = minority_targets[neighbor_idx]
        
        synthetic_sample = sample + np.random.rand() * (neighbor_sample - sample)
        synthetic_target = target + np.random.rand() * (neighbor_target - target)
        
        synthetic_samples.append(synthetic_sample)
        synthetic_targets.append(synthetic_target)
    
    # Combine original and synthetic samples
    X_res = np.vstack((X, np.array(synthetic_samples)))
    y_res = np.hstack((y, np.array(synthetic_targets)))
    
    return X_res, y_res

# Applying state_type data
np.random.seed(42)
X = np.random.rand(100, 2)
y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(100) * 0.1
state_type = np.array(['Blue'] * 60 + ['Red'] * 30 + ['Swing'] * 10)

# Split the data
X_train, X_test, y_train, y_test, state_type_train, state_type_test = train_test_split(
    X, y, state_type, test_size=0.3, random_state=42)

# Apply custom SMOTER for categorical minority group
X_res, y_res = smoter_categorical(X_train, y_train, state_type_train, 'Blue', k_neighbors=5, new_samples=50)

# Train the model on resampled data
model = LinearRegression()
model.fit(X_res, y_res)

# Predict the target variable for the test data
y_pred = model.predict(X_test)

# Create a DataFrame containing true target values, predicted values, and state types
y_test_df = pd.DataFrame({'cases_per_100k': y_test, 'predicted': y_pred, 'state_type': state_type_test})

# Calculate RMSE per state type
rmse_by_state_type = y_test_df.groupby('state_type').apply(lambda x: np.sqrt(mean_squared_error(x['cases_per_100k'], x['predicted']))).reset_index()
rmse_by_state_type.columns = ['State Type', 'RMSE']

# Plot RMSE by state type
plt.figure(figsize=(10, 6))
sns.barplot(x='State Type', y='RMSE', data=rmse_by_state_type)
plt.title('RMSE by State Type')
plt.xlabel('State Type')
plt.ylabel('RMSE')
plt.show()

# ANOVA F-value and p-value
f_value, p_value = f_oneway(df[df['state_type'] == 'Blue']['cases_per_100k'],
                             df[df['state_type'] == 'Red']['cases_per_100k'],
                             df[df['state_type'] == 'Swing']['cases_per_100k'])
print("ANOVA F-value:", f_value)
print("p-value:", p_value)

# Demographic Parity
demographic_parity = df.groupby('state_type')['cases_per_100k'].mean().reset_index()
demographic_parity.columns = ['State Type', 'Mean Predicted Value']
print("\nDemographic Parity:\n", demographic_parity)

# Equalized Odds (Residuals)
# Calculate residuals
residuals = y_test - y_pred

# Create a DataFrame with residuals and state_type
results_df = pd.DataFrame({'Residuals': residuals, 'state_type': state_type_test})

# Group by state_type and calculate mean and standard deviation of residuals
residuals_by_state_type = results_df.groupby('state_type').agg({'Residuals': [list, np.mean, np.std]}).reset_index()

# Rename columns for clarity
residuals_by_state_type.columns = ['State_Type', 'Residuals', 'Mean Residual', 'Std Residual']

print("\nEqualized Odds (Residuals):")
print(residuals_by_state_type[['State_Type', 'Mean Residual', 'Std Residual']])

# Predictive Parity (Mean Absolute Error)
mae_by_state_type = y_test_df.groupby('state_type').apply(lambda x: mean_absolute_error(x['cases_per_100k'], x['predicted'])).reset_index()
mae_by_state_type.columns = ['State Type', 'MAE']
print("\nPredictive Parity (Mean Absolute Error):")
print(mae_by_state_type)

# Data Points Count by State Type
data_points_count = df['state_type'].value_counts().reset_index()
data_points_count.columns = ['State Type', 'Counts']
print("\nData Points Count by State Type:\n", data_points_count)
