# -*- coding: utf-8 -*-
"""yield_spread_neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kA3xHh-5MzKCs5cbr9Zl3R40aahg6H_j
"""

# initialization code
from google.colab import drive
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# path to drive where csv file is read from
drive.mount('/content/drive')
datadir = "/content/drive/MyDrive/BBGdatasets/sampleDailyInputData_12-27-21.csv"

# load historical data from path into a pandas DataFrame
df = pd.read_csv(datadir, index_col=0, header=[0,1], parse_dates=True)
df.columns = [' '.join(col).strip() for col in df.columns.values]

# transform existing DataFrame by calculating yield spreads and appending to end
df['yield_spread_MTGEFNCL_Index_LRC30APR_Index'] = df['MTGEFNCL Index PX_LAST'] - df['LRC30APR Index PX_LAST']
df['yield_spread_MTGEFNCL_Index_GT5_Govt'] = df['MTGEFNCL Index PX_LAST'] - df['GT5 Govt PX_LAST']
df['yield_spread_MTGEFNCL_Index_USGG5YR_Index'] = df['MTGEFNCL Index PX_LAST'] - df['USGG5YR Index PX_LAST']
df['yield_spread_LRC30APR_Index_MTGEFNCL_Index'] = df['LRC30APR Index PX_LAST'] - df['MTGEFNCL Index PX_LAST']
df['yield_spread_LRC30APR_Index_GT5_Govt'] = df['LRC30APR Index PX_LAST'] - df['GT5 Govt PX_LAST']
df['yield_spread_LRC30APR_Index_USGG5YR_Index'] = df['LRC30APR Index PX_LAST'] - df['USGG5YR Index PX_LAST']
df['yield_spread_GT5_Govt_MTGEFNCL_Index'] = df['GT5 Govt PX_LAST'] - df['MTGEFNCL Index PX_LAST']
df['yield_spread_GT5_Govt_LRC30APR_Index'] = df['GT5 Govt PX_LAST'] - df['LRC30APR Index PX_LAST']
df['yield_spread_GT5_Govt_USGG5YR Index'] = df['GT5 Govt PX_LAST'] - df['USGG5YR Index PX_LAST']
df['yield_spread_USGG5YR_Index_MTGEFNCL_Index'] = df['USGG5YR Index PX_LAST'] - df['MTGEFNCL Index PX_LAST']
df['yield_spread_USGG5YR_Index_LRC30APR_Index'] = df['USGG5YR Index PX_LAST'] - df['LRC30APR Index PX_LAST']
df['yield_spread_USGG5YR_Index_GT5_Govt'] = df['USGG5YR Index PX_LAST'] - df['GT5 Govt PX_LAST']

# number of yield spreads, adjust accordingly
n = 12

# take yield spreads and other market information
yield_spreads = df.iloc[:, -n:]
other_columns = df.iloc[:, :-n]

# concatenate yield spreads with other market information to restructure into new DataFrame
df_new = pd.concat([yield_spreads, other_columns], axis=1)

# assign the new dataframe to df to not overwrite original DataFrame as well as drop empty columns
data = df_new.dropna(axis=1)

# separate the yield spreads which are our targets from other market information again
yield_spreads = data.iloc[:, :n].values
market_info = data.iloc[:, n:].values

# device configuration for using Nvidia GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert the data to PyTorch tensors of type torch.float32 and run on GPU if available
X_yield_spreads = torch.tensor(yield_spreads, dtype=torch.float32).to(device)
X_market_info = torch.tensor(market_info, dtype=torch.float32).to(device)

# concatenate the yield spreads and other market information tensors as well as run on GPU if available
X = torch.cat((X_yield_spreads, X_market_info), dim=1).to(device)
y = torch.tensor(yield_spreads, dtype=torch.float32).to(device)

# split the data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

# note that no scalers are needed as features are similar

# define the neural network model named Spread
class Spread(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output):
        super(Spread, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# hyperparameters to control learning proccess of model
input_size = data.shape[1]
hidden_size_1 = 264
hidden_size_2 = 132
output = n
num_epochs = input_size * 11
learning_rate = 0.01

# instantiating the neural network model with GPU if available
model = Spread(input_size, hidden_size_1, hidden_size_2, output).to(device)

# define the Mean Squred Error loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop of model
for epoch in range(num_epochs):
    model.train()

    # convert x_train and y_train to run on GPU if available
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # update learnable weights of model as well as output input of x_train set
    optimizer.zero_grad()
    outputs = model(x_train)

    # MSE loss between x_train outputs and actual y_train values
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # evaluate model on validation set without calculating gradients
    model.eval()
    with torch.no_grad():
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val)

    # display current Train Loss and Val Loss for each epoch
    print(f"Epoch {epoch+1}: Train Loss = {loss.item()}, Val Loss = {val_loss.item()}")

# save only state_dict of model and where
save_model_name = 'spread.pt'
PATH = f"/content/drive/MyDrive/BBGdatasets/MLmodels/{save_model_name}"

torch.save(model.state_dict(), PATH)

# load loaded dictionary of model
model = Spread(input_size, hidden_size_1, hidden_size_2, output).to(device)
model.load_state_dict(torch.load(PATH))

# evaluate model on test set similar to validation set
model.eval()
with torch.no_grad():
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    test_outputs = model(x_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item()}")

# convert the predicted tensor back to a NumPy array
predictions = test_outputs.cpu().numpy()

# calculate the average predicted yield spread for each instrument
avg_predicted_spreads = np.mean(predictions, axis=0)

# get instrument names
instrument_names = data.columns[:n]

# create a list of tuples with instrument names and their average predicted spreads
instrument_spreads = list(zip(instrument_names, avg_predicted_spreads))

# rank all of the instruments based on their average predicted yield spreads
ranked_instruments = sorted(instrument_spreads, key=lambda x: x[1], reverse=True)

# print instrument rankings
print("Instrument Rankings:")
for rank, (instrument_name, spread) in enumerate(ranked_instruments):
    print(f"Rank {rank+1}: Instrument {instrument_name} ({spread})")

