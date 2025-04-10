#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import optuna
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import re


# In[5]:


train = pd.read_csv("metallurgica2025/train.csv")
test = pd.read_csv("metallurgica2025/test.csv")


# In[4]:


print(torch.cuda.is_available())


# # #DATA_PREPROCESSING

# In[6]:


train.shape


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.head(15)


# In[10]:


train["Alloy class"].unique()


# In[11]:


train.describe().iloc[:,:20]


# In[12]:


train.isnull().sum()


# In[13]:


test.isnull().sum()


# In[14]:


null_row = {}
for column in train.columns:
    temp = list(train[train[column].isnull()].index)
    if len(temp):
        null_row[column] = temp
        
common_indices = set.intersection(*[set(indices) for indices in null_row.values()])
common_indices


# In[15]:


train = train.drop('ID', axis=1)
train.dropna(subset="Electrical conductivity (%IACS)", inplace=True)
test = test.drop('ID', axis=1)
train = train.drop(["Yield strength (MPa)","Ultimate tensile strength (MPa)","Alloy formula", "Alloy class"], axis=1)
test = test.drop(["Yield strength (MPa)","Ultimate tensile strength (MPa)"], axis=1)


# In[16]:


le = LabelEncoder()


# In[17]:


def label_encoder(encoder, data):
    for clm  in data.select_dtypes(include="object").columns:
        data[clm] = encoder.fit_transform(data[clm])
    return data


# In[18]:


train = label_encoder(le, train)
test = label_encoder(le, test)


# In[19]:


def fill_null(list_of_clm, data):
    for clmn in list_of_clm:
        if data[clmn].dtype == 'object':
            data[clmn] = data[clmn].fillna("NaN")
            
        if data[clmn].dtype == 'float64':
            data[clmn] = data[clmn].fillna(train[clmn].mean())
            
        if data[clmn].dtype == 'int32':
            data[clmn] = data[clmn].fillna(train[clmn].mode()[0])
    return data


# In[20]:


clmns = train.columns
train = fill_null(clmns, train)


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], test_size=0.2, random_state=42)


# # #DATA VISUALIZATION

# In[22]:


sns.set(style="whitegrid")


# In[23]:


plt.figure(figsize=(8, 5))
sns.histplot(train["Electrical conductivity (%IACS)"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Electrical Conductivity (%IACS)")
plt.xlabel("Electrical Conductivity (%IACS)")
plt.ylabel("Frequency")
plt.show()


# In[24]:


plt.figure(figsize=(15, 12))
corr_matrix = train.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()


# In[25]:


selected_features = ['Tss (K)', 'tss (h)', 'CR reduction (%)','Aging', 'Tag (K)', 'tag (h)', 'Secondary thermo-mechanical process','Hardness (HV)','Electrical conductivity (%IACS)']
sns.pairplot(train[selected_features], diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=train[selected_features], orient="h", palette="Set2")
plt.title("Boxplot of Selected Features")
plt.show()


# # #MODEL SELECTION
# 

# ## ->tranditional ML models

# In[27]:


models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0)
}


# In[28]:


results = {}
for name, model in models.items():   
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)   
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "R2": r2}


# In[29]:


results_df = pd.DataFrame(results).T
print(results_df.sort_values(by="R2", ascending=False))


# ## ->deep learning architecture_1

# In[30]:


scaler = StandardScaler()


# In[31]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[32]:


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# In[33]:


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[34]:


'''class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout_rate):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, 1)
        )
    
    def forward(self, x):
        return self.model(x)'''


# In[35]:


input_dim = X_train_tensor.shape[1] 
output_dim = 1 

class RM(nn.Module):
    def __init__(self, hidden1, hidden2, dropout_rate):
        super(RM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # Batch Normalization
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# In[36]:


hidden1 = 128  
hidden2 = 64   
dropout_rate = 0.2
input_size = X_train_tensor.shape[1]


# In[37]:


model_RM = RM(hidden1, hidden2, dropout_rate)


# In[38]:


criterion = nn.MSELoss()
optimizer = optim.AdamW(model_RM.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# In[39]:


epochs = 150
for epoch in range(epochs):
    model_RM.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model_RM(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


# In[40]:


model_RM.eval()
with torch.no_grad():
    y_pred_tensor = model_RM(X_test_tensor).detach().numpy().flatten()
mae = mean_absolute_error(y_test, y_pred_tensor)
mse = mean_squared_error(y_test, y_pred_tensor)
r2 = r2_score(y_test, y_pred_tensor)


# In[41]:


print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# In[42]:


residuals = np.abs(y_test.values.flatten() - y_pred_tensor)
plt.hist(residuals, bins=30, color='skyblue', edgecolor='k')
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Distribution of Absolute Errors (MAE)")
plt.show()


# ## WIDEANDDEEP

# In[45]:


class WideAndDeep(nn.Module):
    def __init__(self, input_size, deep_hidden_sizes, dropout_rate=0.2):
        super(WideAndDeep, self).__init__()
        self.wide = nn.Linear(input_size, 1)
        layers = []
        current_input = input_size
        for hidden_size in deep_hidden_sizes:
            layers.append(nn.Linear(current_input, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_input = hidden_size
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(current_input, 1)
        
    def forward(self, x):
        wide_out = self.wide(x)
        deep_features = self.deep(x)
        deep_out = self.deep_out(deep_features)
        return wide_out + deep_out


# In[46]:


input_size = X_train_tensor.shape[1] 
model_WND = WideAndDeep(input_size=input_size, deep_hidden_sizes=[128, 64], dropout_rate=0.2)


# In[49]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model_WND.parameters(), lr=0.001)


# In[50]:


epochs = 100
for epoch in range(epochs):
    model_WND.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_WND(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# In[51]:


model_WND.eval()
with torch.no_grad():
    y_pred_tensor = model_WND(X_test_tensor).detach().numpy().flatten()
mae = mean_absolute_error(y_test, y_pred_tensor)
mse = mean_squared_error(y_test, y_pred_tensor)
r2 = r2_score(y_test, y_pred_tensor)


# In[52]:


print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# # HYPERPARAM TUNNING OF DL
# 

# In[65]:


X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)


# In[66]:


train_dataset_sub = TensorDataset(X_train_sub, y_train_sub)
train_loader_sub = DataLoader(train_dataset_sub, batch_size=32, shuffle=True)


# In[67]:


val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# In[72]:


def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 64, 512)
    hidden2 = trial.suggest_int("hidden2", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])    
    input_size = X_train_tensor.shape[1]
    model = RM( hidden1=hidden1, hidden2=hidden2, dropout_rate=dropout_rate)   
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)   
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader_sub:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)   
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val).detach().numpy().flatten()
    val_mae = mean_absolute_error(y_val.numpy(), y_pred_val)
    return val_mae


# In[73]:


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)


# In[74]:


def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    criterion = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # **Validation Phase**
    model.eval()
    mae_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mae_loss += criterion(outputs, targets).item()
    
    return mae_loss / len(val_loader)


# In[77]:


def objective_(trial):
    hidden1 = trial.suggest_int("hidden1", 50, 200)
    hidden2 = trial.suggest_int("hidden2", 50, 200)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = RM(hidden1, hidden2, dropout_rate)  # Define your model
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # **Use your existing train-test split**
    train_dataset_sub = TensorDataset(X_train_sub, y_train_sub)
    val_dataset = TensorDataset(X_val, y_val)  # Validation dataset

    train_loader_sub = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Validation DataLoader

    # Train and evaluate
    mae = train_and_evaluate(model, optimizer, train_loader_sub, val_loader)

    return mae


# In[78]:


study = optuna.create_study(direction="minimize")
study.optimize(objective_, n_trials=50)


# In[279]:


print("Best trial:", study.best_trial)


# # HYPERPARAM TUNNING OF ML MODELS

# In[283]:


param_grids = {
    "Random Forest": {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 30),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
    },
    "Ridge Regression": {
        "alpha": uniform(0.01, 10),
    },
    "LightGBM": {
        "num_leaves": randint(20, 100),
        "max_depth": randint(3, 20),
        "learning_rate": uniform(0.01, 0.2),
        "n_estimators": randint(50, 300),
    },
    "XGBoost": {
        "max_depth": randint(3, 20),
        "learning_rate": uniform(0.01, 0.3),
        "n_estimators": randint(50, 300),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
    }
}


# In[293]:


best_models = {}
results = []

for name, model in models.items():
    print(f"Tuning {name}...")

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[name],
        n_iter=30,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train_1, y_train_1)

    best_models[name] = search.best_estimator_

    y_pred = search.predict(X_test_1)
    mae = mean_absolute_error(y_test_1, y_pred)
    mse = mean_squared_error(y_test_1, y_pred)
    r2 = r2_score(y_test_1, y_pred)
    results.append({"Model": name, "MAE": mae, "MSE": mse, "R2": r2})


# In[294]:


results_df = pd.DataFrame(results)
print(results_df)


# In[298]:


best_models


# In[299]:


#initializing best model
model_RF = RandomForestRegressor(max_depth=22, min_samples_leaf=9, min_samples_split=4,
                       n_estimators=278, random_state=42)


# In[363]:


model_RF.fit(X_train_1, y_train_1)


# # HYPERPARAM TUNNING OF WIDE_AND_DEEP

# In[106]:


train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
from torch.utils.data import Subset
train_dataset_sub = Subset(train_dataset, train_indices)
val_dataset = Subset(train_dataset, val_indices)


# In[110]:


def objective_WND(trial):s
    hidden1 = trial.suggest_int("hidden1", 64, 256, step=32)
    hidden2 = trial.suggest_int("hidden2", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    model = WideAndDeep(input_size=X_train_tensor.shape[1],deep_hidden_sizes=[hidden1, hidden2],dropout_rate=dropout_rate)
    model.train()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_loader_local = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    val_loader_local = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 150  
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_X, batch_y in train_loader_local:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader_local:
            outputs = model(batch_X)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae

from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
from torch.utils.data import Subset
train_dataset_sub = Subset(train_dataset, train_indices)
val_dataset = Subset(train_dataset, val_indices)

study = optuna.create_study(direction="minimize")
study.optimize(objective_WND, n_trials=20)
print("Best hyperparameters:", study.best_params)


# In[111]:


best_params = study.best_trial.params


# In[112]:


model_WND = WideAndDeep(input_size=input_size,deep_hidden_sizes=[best_params["hidden1"], best_params["hidden2"]],dropout_rate=best_params["dropout_rate"])
criterion = nn.MSELoss()
optimizer = optim.Adam(model_WND.parameters(),lr=best_params["lr"],weight_decay=best_params["weight_decay"])
train_loader_best = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
test_loader_best = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

epochs = 100
for epoch in range(epochs):
    model_WND.train()
    total_loss = 0
    for batch_X, batch_y in train_loader_best:
        optimizer.zero_grad()
        outputs = model_WND(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader_best)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

model_WND.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader_best:
        outputs = model_WND(batch_X)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())

final_mae = mean_absolute_error(y_true, y_pred)
final_mse = mean_squared_error(y_true, y_pred)
final_r2 = r2_score(y_true, y_pred)

print(f"\nFinal MAE: {final_mae:.4f}")
print(f"Final MSE: {final_mse:.4f}")
print(f"Final R2 Score: {final_r2:.4f}")


# # ENSEMBLE LEARNING

# In[89]:


model_RM = RM(
    hidden1=60,
    hidden2=166,
    dropout_rate=0.12171071371739912
)
optimizer_rm = optim.Adam(model_RM.parameters(), lr=0.0007552947355857924)
model_RM.eval()
with torch.no_grad():
    nn_pred = model_RM(X_test_tensor).cpu().numpy().flatten()

model_WND = WideAndDeep(
    input_size,
    deep_hidden_sizes=[256, 96],
    dropout_rate=0.10110057906299971
)
optimizer_wnd = optim.Adam(model_WND.parameters(), lr=0.004296679753904131, weight_decay=0.0008084908961173071)
model_WND.eval()
with torch.no_grad():
    wnd_pred = model_WND(X_test_tensor).cpu().numpy().flatten()


# In[90]:


ensemble_pred = ( nn_pred + wnd_pred) / 3.0


# In[91]:


ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)


# In[92]:


print(f"Ensemble MAE: {ensemble_mae:.4f}")
print(f"Ensemble MSE: {ensemble_mse:.4f}")
print(f"Ensemble R2 Score: {ensemble_r2:.4f}")


# In[113]:


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Linear(features, features),
            nn.BatchNorm1d(features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ResNetRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, dropout_rate=0.2):
        super(ResNetRegressor, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.res_blocks(out)
        out = self.fc_out(out)
        return out


# # RESNET

# In[114]:


def objective_resnet(trial):
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    num_blocks = trial.suggest_int("num_blocks", 1, 4)  # Number of residual blocks
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = ResNetRegressor(input_size=X_train_tensor.shape[1],
                            hidden_size=hidden_size,
                            num_blocks=num_blocks,
                            dropout_rate=dropout_rate)
    model.train()

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)

    train_dataset_sub = Subset(train_dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    
    train_loader_local = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    val_loader_local = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 150
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_X, batch_y in train_loader_local:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader_local:
            outputs = model(batch_X)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae
study_resnet = optuna.create_study(direction="minimize")
study_resnet.optimize(objective_resnet, n_trials=30)
print("Best hyperparameters for ResNetRegressor:", study_resnet.best_params)


# # TabNet

# In[47]:


X_train_np = X_train_scaled  
y_train_np = y_train.values.reshape(-1, 1)
X_test_np = X_test_scaled    # scaled test data
y_test_np = y_test.values.reshape(-1, 1)


# In[49]:


get_ipython().system('pip install pytorch_tabnet')


# In[50]:


from pytorch_tabnet.tab_model import TabNetRegressor


# In[51]:


tabnet_params = {
    "n_d": 64,               # dimension of the decision prediction layer
    "n_a": 64,               # dimension of the attention embedding
    "n_steps": 5,            # number of steps in the architecture
    "gamma": 1.5,            # relaxation factor for feature reusage
    "lambda_sparse": 1e-3,   # sparsity regularization
    "optimizer_fn": optim.Adam,
    "optimizer_params": dict(lr=1e-3, weight_decay=1e-5),
    "mask_type": "entmax"    # use "sparsemax" if desired
}


# In[52]:


model_tabnet = TabNetRegressor(**tabnet_params)


# In[126]:


model_tabnet.fit(
    X_train_np, y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_metric=['mae'],
    max_epochs=100,
    patience=10,
    batch_size=256,         
    virtual_batch_size=128, 
    num_workers=0,
    drop_last=False
)


# In[127]:


preds = model_tabnet.predict(X_test_np)
mae_tabnet = mean_absolute_error(y_test_np, preds)
print("TabNet MAE:", mae_tabnet)


# # trying WND again

# In[53]:


def objective_WND(trial):
    # Hyperparameter suggestions (allowing larger capacity)
    hidden1 = trial.suggest_int("hidden1", 128, 512, step=32)
    hidden2 = trial.suggest_int("hidden2", 64, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Instantiate the Wide & Deep model with current trial parameters
    model = WideAndDeep(
        input_size=X_train_tensor.shape[1],
        deep_hidden_sizes=[hidden1, hidden2],
        dropout_rate=dropout_rate
    )
    model.train()
    
    # Use SmoothL1Loss (Huber loss) to robustly optimize MAE
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create data loaders with current batch size for training and validation
    train_loader_local = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    val_loader_local = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    max_epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20  # Stop if no improvement for 20 epochs
    
    # Training loop with early stopping
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader_local:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader_local)
        scheduler.step(avg_loss)
        
        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader_local:
                outputs = model(batch_X)
                loss_val = criterion(outputs, batch_y)
                val_losses.append(loss_val.item())
        val_loss = np.mean(val_losses)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                # Uncomment the line below to print early stopping info if desired:
                # print(f"Early stopping at epoch {epoch+1} with val_loss {val_loss:.4f}")
                break
    
    # Evaluate final model on validation set (using MAE)
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader_local:
            outputs = model(batch_X)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae


# In[54]:


from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
from torch.utils.data import Subset
train_dataset_sub = Subset(train_dataset, train_indices)
val_dataset = Subset(train_dataset, val_indices)


# In[55]:


study = optuna.create_study(direction="minimize")
study.optimize(objective_WND, n_trials=30)
print("Best hyperparameters:", study.best_params)


# In[90]:


best_params = {
    'hidden1': 224,
    'hidden2': 64,
    'dropout_rate': 0.17469366441900647,
    'lr': 4.830921562807099e-05,
    'weight_decay': 9.321023743395316e-05,
    'batch_size': 32
}


# In[91]:


input_size = X_train_tensor.shape[1]
model_WND = WideAndDeep(
    input_size=input_size,
    deep_hidden_sizes=[best_params['hidden1'], best_params['hidden2']],
    dropout_rate=best_params['dropout_rate']
)


# In[92]:


criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model_WND.parameters(), lr=best_params["lr"], weight_decay=best_params['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


# In[93]:


train_loader_best = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader_best = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)


# In[94]:


epochs = 150
for epoch in range(epochs):
    model_WND.train()
    total_loss = 0
    for batch_X, batch_y in train_loader_best:
        optimizer.zero_grad()
        outputs = model_WND(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader_best):.4f}")


# In[95]:


model_WND.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader_best:
        outputs = model_WND(batch_X)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())


# In[96]:


final_mae = mean_absolute_error(y_true, y_pred)
final_mse = mean_squared_error(y_true, y_pred)
final_r2 = r2_score(y_true, y_pred)

print(f"\nFinal MAE: {final_mae:.4f}")
print(f"Final MSE: {final_mse:.4f}")
print(f"Final R2 Score: {final_r2:.4f}")


# # PREDICTION

# In[99]:


best_models


# In[100]:


best_rf_model = best_models["Random Forest"]


# In[101]:


best_rf_model.fit(X_train, y_train)


# In[106]:


test = test.drop(["Alloy formula","Alloy class"], axis=1)


# In[111]:


clmns = test.columns
test = fill_null(clmns, test)


# In[112]:


test_predictions = best_rf_model.predict(test)


# In[113]:


test["Predicted Electrical conductivity (%IACS)"] = test_predictions


# In[114]:


print(test.head())


# In[115]:


y_train


# In[117]:


test_original = pd.read_csv("metallurgica2025/test.csv")


# In[119]:


output_df = pd.DataFrame({
    "ID": test_original["ID"],
    "Electrical conductivity (%IACS)": test_predictions
})


# In[120]:


output_df.to_csv("predictions.csv", index=False)


# In[124]:


p = pd.read_csv("predictions.csv")
p.head()

