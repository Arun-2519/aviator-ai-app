import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Color mapping function
def color_cat(x):
    if 1 <= x < 2:
        return 0  # blue
    elif 2 <= x < 10:
        return 1  # dark blue
    elif x >= 10:
        return 2  # pink
    return -1

def color_str(code):
    return {0: 'blue', 1: 'dark blue', 2: 'pink'}.get(code, 'unknown')

# 2. Feature engineering for windowed supervised learning
def extract_streaks(colors):
    streaks = []
    current_streak = 1
    streak_color = colors[0]
    max_streak = 1

    for c in colors[1:]:
        if c == streak_color:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
            streak_color = c
        streaks.append(current_streak)
    return max_streak

def create_rolling_features(df, N=10):
    # Numeric features
    rolled = df['Crash Multiplier'].rolling(N)
    df['mean'] = rolled.mean().shift(1)
    df['median'] = rolled.median().shift(1)
    df['std'] = rolled.std().shift(1)
    df['min'] = rolled.min().shift(1)
    df['max'] = rolled.max().shift(1)
    # Difference last-to-prev
    df['diff_last'] = df['Crash Multiplier'].diff().shift(-1)
    # Ratio last-to-prev
    df['ratio_last'] = df['Crash Multiplier'].shift(1) / df['Crash Multiplier'].shift(2)
    df['ColorCode'] = df['Crash Multiplier'].map(color_cat)
    features = []
    targets = []
    for i in range(N, len(df) - 1):
        window_multipliers = df['Crash Multiplier'].iloc[i-N:i]
        window_colors = df['ColorCode'].iloc[i-N:i].values
        streak = (window_colors == window_colors[0]).sum()
        pink_count = (window_colors == 2).sum()
        alternations = np.sum(window_colors[1:] != window_colors[:-1])
        features.append(
            list(window_multipliers)
            + [df['mean'][i], df['median'][i], df['std'][i], df['min'][i], df['max'][i], df['diff_last'][i], df['ratio_last'][i]]
            + list(window_colors)
            + [streak, pink_count, alternations]
        )
        targets.append(df['ColorCode'].iloc[i])
    return np.array(features), np.array(targets)

# 3. Load data and generate features
def prepare_ml_data(csv_path, N=10):
    df = pd.read_csv(csv_path)
    X, y = create_rolling_features(df, N=N)
    return X, y

# 4. Model and evaluation functions
def train_baseline_rf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    # Example: simulate dataset if no CSV available
    np.random.seed(42)
    arr = np.random.choice(
        [1.1, 1.5, 1.7, 2.3, 2.6, 9.9, 10.5, 17.3, 22.1, 50.4], size=200, replace=True
    )
    df = pd.DataFrame({'Crash Multiplier': arr})
    df.to_csv("aviator_crash_history.csv", index=False)
    X, y = create_rolling_features(df, N=10)
    # 5. ML split, fit, evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    clf = train_baseline_rf(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['blue','dark blue','pink']))

# ---------- DEEP LEARNING PIPELINE: LSTM (PyTorch) ----------
# You can run this section in a separate deep learning notebook.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# Prepare data
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
BATCH_SIZE = 32
train_ds = TensorDataset(X, y)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# Define model
class AviatorLSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))
        return self.fc(x[:,-1,:])
# Training and usage shown in pipeline above.
'''
