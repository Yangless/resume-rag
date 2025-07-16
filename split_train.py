import pandas as pd
df = pd.read_csv('processed_output1.csv')
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('train.csv', index=False)
val_df.to_csv('eval.csv', index=False)