import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('video_engagement_data.csv')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
train_data['engagement'] = scaler.fit_transform(train_data['engagement'].values.reshape(-1, 1))
test_data['engagement'] = scaler.transform(test_data['engagement'].values.reshape(-1, 1))

# Save preprocessed data
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
