import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the data
df = pd.read_csv('processed_data/100k_population_data.csv', low_memory=False)

# Drop rows with NaN values
df.dropna(inplace=True)

# Dropping unneeded columns
dropping_columns = ['target_end_date']
df.drop(columns=dropping_columns, inplace=True)

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

# Apply the function to create the state_type column
df['state_type'] = df.apply(get_state_type, axis=1)

# Dropping unneeded columns
dropping_columns = ['location_key', 'location', 'blue_states', 'red_states', 'swing_states', 'unemployment_rate', 'hospitalized_per_100k']
df.drop(columns=dropping_columns, inplace=True)

# Display final dataframe
display(df)

from imblearn.combine import SMOTEENN

# Encoding state_type column for performing Random Forest after
def encode_state_type(state):
    if state == 'Blue':
        return 0
    elif state == 'Red':
        return 1
    else: 
        return 2 # For Swing states

df['state_type_encoded'] = df['state_type'].apply(encode_state_type)

# Drop the original 'state_type' column
df.drop(columns=['state_type'], inplace=True)

# Split data into features (X) and target variable (y)
X = df.drop(columns=['state_type_encoded'])
y = df['state_type_encoded']  

# Initialize SMOTEENN for over- and under-sampling
smote_enn = SMOTEENN()

# Apply SMOTEENN to the data
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Convert y_resampled to a DataFrame and count the number of states in each type
y_resampled_df = pd.DataFrame(y_resampled, columns=['state_type_encoded'])

# Print the number of red states, blue states, and swing states
print(y_resampled_df['state_type_encoded'].value_counts())

# Display new dataframe
display(df)

# Split the resampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Cross-validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# Print evaluation metrics
print(f'R2: {r2:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Cross-Validation RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})')
