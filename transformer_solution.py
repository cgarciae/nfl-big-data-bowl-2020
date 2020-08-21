#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl 2020 - 1st place solution The Zoo

# This notebook aims to reproduce the NFL Big Data Bowl 2020 winner solution described in [1]. The purpose of the competiton was to develop a model to predict how many yards a team will gain on given rushing plays as they happen [2]. The dataset contains game, play, and player-level data. This elegant solution is only based on player-level data. In particular, on relative location and speed features only.
# 
# To understand the proposed solution, assume that in a simplified definition, a rushing play consists on:
# - A rusher, whose aim is to run forward as far as possible
# - 11 defense players who are trying to stop the rusher
# - 10 remaining offense players trying to prevent defenders from blocking or tackling the rusher
# 
# Based on this simplified version of the game, the authors in [1] came up with the following network structure:
# 
# <img src="images/model_structure.png" style="width:680px;height:200px;">
# 
# 
# We will go into the details throughout this notebook.
# 
# The remainder of this notebook is organized as follows. Section 1 describes and contains the code for data processing and data augmentation. Section 2 provides the model structure. Finally, section 3 draws some conclusions and some possible improvements

# ## Data Processing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import utils


# In[2]:


train = pd.read_csv('data/train.csv', dtype={'WindSpeed': 'object'})


# First, we divide the dataset into two daframes. The first dataframe (df_players) contains the columns related to the player-level features. Meanwhile, the second dataframe (df_play) is formed by some play-level features which will be useful to perform some transformations on df_players.

# In[89]:


def split_play_and_player_cols(df, predicting=False):
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    
    df['PlayId'] = df['PlayId'].astype(str)
    
    # We must assume here that the first 22 rows correspond to the same player:
    player_cols = [
        'PlayId', # This is the link between them
        'Season',
        'Team',
        'X',
        'Y',
        'S',
        'Dis',
        'Dir',
        'NflId',
        'IsRusher',
        'Yards'
    ]

    df_players = df[player_cols]
    
    play_cols = [
        'PlayId',
        'Season',
        'PossessionTeam',
        'HomeTeamAbbr',
        'VisitorTeamAbbr',
        'PlayDirection', 
        'FieldPosition',
        'YardLine',
    ]
    
    if not predicting:
        play_cols.append('Yards')
        
    df_play = df[play_cols].copy()

    ## Fillna in FieldPosition attribute
    #df['FieldPosition'] = df.groupby(['PlayId'], sort=False)['FieldPosition'].apply(lambda x: x.ffill().bfill())
    
    # Get first 
    df_play = df_play.groupby('PlayId').first().reset_index()

    print('rows/plays in df: ', len(df_play))
    assert df_play.PlayId.nunique() == df.PlayId.nunique(), "Play/player split failed?"  # Boom
    
    return df_play, df_players

play_ids = train["PlayId"].unique()

df_play, df_players = split_play_and_player_cols(train)


# We have some problems with the enconding of the teams such as BLT and BAL or ARZ and ARI. Let's fix it.

# In[90]:


def process_team_abbr(df):

    #These are only problems:
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in train['PossessionTeam'].unique():
        map_abbr[abb] = abb

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossession'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    
    return

process_team_abbr(df_play)


# In[91]:


def process_play_direction(df):
    df['IsPlayLeftToRight'] = df['PlayDirection'].apply(lambda val: True if val.strip() == 'right' else False)
    return

process_play_direction(df_play)


# We compute how many yards are left to the end-zone.

# In[92]:


def process_yard_til_end_zone(df):
    def convert_to_yardline100(row):
        return (100 - row['YardLine']) if (row['PossessionTeam'] == row['FieldPosition']) else row['YardLine']
    df['Yardline100'] = df.apply(convert_to_yardline100, axis=1)
    return

process_yard_til_end_zone(df_play)


# Now, we add the computed features to df_players

# In[93]:


df_players = df_players.merge(
    df_play[['PlayId', 'PossessionTeam', 'HomeTeamAbbr', 'PlayDirection', 'Yardline100']], 
    how='left', on='PlayId')


# In[94]:


df_players.loc[df_players.Season == 2017].plot.scatter(x='Dis', y='S', title='Season 2017',grid=True)


# In[95]:


df_players.loc[df_players.Season == 2018].plot.scatter(x='Dis', y='S', title='Season 2018', grid=True)


# In 2018 data we can see that S is linearly related to Dis. However, data in 2017 is not very fit. Using a linear regresion to fit the 2018 data, we found that S can be replaced by 10*Dir. This give an improvment in the predictions

# In[96]:


X = df_players.loc[df_players.Season == 2018]['Dis']
y = df_players.loc[df_players.Season == 2018]['S']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit() 
model.summary()


# In[97]:


df_players.loc[df_players.Season == 2017, 'S'] = 10*df_players.loc[df_players.Season == 2017,'Dis']


# Now, let's adjusted the data to always be from left to right.

# In[98]:


def standarize_direction(df):
    # adjusted the data to always be from left to right
    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0

    df['ToLeft'] = df.PlayDirection == "left"
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']
    df['Dir_std'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
   
    #Replace Null in Dir_rad
    df.loc[(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = 0.0
    df.loc[~(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = np.pi

standarize_direction(df_players)


# We adjust only the plays moving to left. To explain the transformation, consider the following images that show two original plays (the purple is the team in offense)

# In[99]:


for play_id in ['20170910001102', '20170910000081']: 
    utils.show_play(play_id, df_players)


# Now, these are the same plays after the transformation

# In[100]:


for play_id in ['20170910001102', '20170910000081']: 
    utils.show_play_std(play_id, df_players)


# Note that we only modify the plays moving to left. The source code to these plots is taken from [3]

# ### Data augmentation
# For training, we assume that in a mirrored world the runs would have had the same outcomes. We apply 50% augmentation to flip the Y coordinates (and all respective relative features emerging from it). Furthermore, the function process_tracking_data computes the projections on X and Y for the velocity of each player and other features relative to rusher.

# In[101]:


def data_augmentation(df, sample_ids):
    df_sample = df.loc[df.PlayId.isin(sample_ids)].copy()
    df_sample['Y_std'] = 160/3  - df_sample['Y_std']
    df_sample['Dir_std'] = df_sample['Dir_std'].apply(lambda x: 2*np.pi - x)
    df_sample['PlayId'] = df_sample['PlayId'].apply(lambda x: x+'_aug')
    return df_sample

def process_tracking_data(df):
    # More feature engineering for all:
    df['Sx'] = df['S']*df['Dir_std'].apply(math.cos)
    df['Sy'] = df['S']*df['Dir_std'].apply(math.sin)
    
    # ball carrier position
    rushers = df[df['IsRusher']].copy()
    rushers.set_index('PlayId', inplace=True, drop=True)
    playId_rusher_map = rushers[['X_std', 'Y_std', 'Sx', 'Sy']].to_dict(orient='index')
    rusher_x = df['PlayId'].apply(lambda val: playId_rusher_map[val]['X_std'])
    rusher_y = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Y_std'])
    rusher_Sx = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sx'])
    rusher_Sy = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sy'])
    
    # Calculate differences between the rusher and the players:
    df['player_minus_rusher_x'] = rusher_x - df['X_std']
    df['player_minus_rusher_y'] = rusher_y - df['Y_std']

    # Velocity parallel to direction of rusher:
    df['player_minus_rusher_Sx'] = rusher_Sx - df['Sx']
    df['player_minus_rusher_Sy'] = rusher_Sy - df['Sy']

    return

sample_ids = np.random.choice(df_play.PlayId.unique(), int(0.5*len(df_play.PlayId.unique())))

df_players_aug = data_augmentation(df_players, sample_ids)
df_players = pd.concat([df_players, df_players_aug])
df_players.reset_index()

df_play_aug = df_play.loc[df_play.PlayId.isin(sample_ids)].copy()
df_play_aug['PlayId'] = df_play_aug['PlayId'].apply(lambda x: x+'_aug')
df_play = pd.concat([df_play, df_play_aug])
df_play.reset_index()

# This is necessary to maintain the order when in the next cell we use groupby
df_players.sort_values(by=['PlayId'],inplace=True)
df_play.sort_values(by=['PlayId'],inplace=True)

process_tracking_data(df_players)


# In[103]:


tracking_level_features = [
    'PlayId',
    'IsOnOffense',
    'X_std',
    'Y_std',
    'Sx',
    'Sy',
    'player_minus_rusher_x',
    'player_minus_rusher_y',
    'player_minus_rusher_Sx',
    'player_minus_rusher_Sy',
    'IsRusher',
    'Yards'
]

df_all_feats = df_players[tracking_level_features]

print('Any null values: ', df_all_feats.isnull().sum().sum())


# Finally, we create the train tensor to feed the convolutional network. The following image depicts the structure of the input tensor:
# 
# <img src="images/input.png" style="width:350px;height:160px;">
# 
# Note that the idea is to reshape the data of a play into a tensor of defense vs offense, using features as channels to apply 2d operations (The figure does not follow the convention on ConvNet, and the channels are in the z-axis). There are 5 vector features which were important (so 10 numeric features if you count projections on X and Y axis). The vectors are relative locations and speeds, so to derive them we used only ‘X’, ‘Y’, ‘S’ and ‘Dir’ variables from data.

# In[104]:


df_all_feats.columns


# In[107]:


from tqdm import tqdm

Xs = []
Ys = []


min_idx_y = 71
max_idx_y = 150


groups = df_all_feats[["PlayId", "Yards", "X_std", "Y_std", "Sx", "Sy", "IsRusher"]].groupby("PlayId")

for play_id, df in tqdm(groups):
    df = df.sort_values("IsRusher", ascending=False)
    Xs.append(df[["X_std", "Y_std", "Sx", "Sy", "IsRusher"]].to_numpy().astype(np.float32))
    
    val = df['Yards'].iloc[0] + 99
    val = min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val
    Ys.append(val)
    

X = np.stack(Xs, axis=0)
Y = np.stack(Ys, axis=0)
print(X.shape, Y.shape)

np.save('data/X_v3(augmented-50).npy', X)
np.save('data/Y_v3(augmented-50).npy', Y)


# In[34]:


get_ipython().run_cell_magic('time', '', '\ngrouped = df_all_feats.groupby(\'PlayId\')\ntrain_x = np.zeros([len(grouped.size()),11,10,10])\ni = 0\nplay_ids = df_play.PlayId.values\nfor name, group in grouped:\n    if name!=play_ids[i]:\n        print("Error")\n\n    [[rusher_x, rusher_y, rusher_Sx, rusher_Sy]] = group.loc[group.IsRusher==1,[\'X_std\', \'Y_std\',\'Sx\',\'Sy\']].values\n\n    offense_ids = group[group.IsOnOffense & ~group.IsRusher].index\n    defense_ids = group[~group.IsOnOffense].index\n\n    for j, defense_id in enumerate(defense_ids):\n        [def_x, def_y, def_Sx, def_Sy] = group.loc[defense_id,[\'X_std\', \'Y_std\',\'Sx\',\'Sy\']].values\n        [def_rusher_x, def_rusher_y] = group.loc[defense_id,[\'player_minus_rusher_x\', \'player_minus_rusher_y\']].values\n        [def_rusher_Sx, def_rusher_Sy] =  group.loc[defense_id,[\'player_minus_rusher_Sx\', \'player_minus_rusher_Sy\']].values\n        \n        train_x[i,j,:,:4] = group.loc[offense_ids,[\'Sx\',\'Sy\',\'X_std\', \'Y_std\']].values - np.array([def_Sx, def_Sy, def_x,def_y])\n        train_x[i,j,:,-6:] = [def_rusher_Sx, def_rusher_Sy, def_rusher_x, def_rusher_y, def_Sx, def_Sy]\n    \n    i+=1\n\nnp.save(\'data/train_x_v3(augmented-50).npy\', train_x)')


# Additionally, for training we clip the target to -30 and 50. 

# In[35]:


# Transform Y into indexed-classes:
train_y = df_play[['PlayId', 'Yards']].copy()

train_y['YardIndex'] = train_y['Yards'].apply(lambda val: val + 99)

min_idx_y = 71
max_idx_y = 150

train_y['YardIndexClipped'] = train_y['YardIndex'].apply(
    lambda val: min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val)

print('max yardIndex: ', train_y.YardIndex.max())
print('max yardIndexClipped: ', train_y.YardIndexClipped.max())
print('min yardIndex: ', train_y.YardIndex.min())
print('min yardIndexClipped: ', train_y.YardIndexClipped.min())

train_y.to_pickle('data/train_y_v3.pkl')


# In[36]:


df_season = df_play[['PlayId', 'Season']].copy()
df_season.to_pickle('data/df_season_v3.pkl')


# ## Train ConvNet

# In[17]:


train_x = np.load('data/X_v3(augmented-50).npy') 
train_y = np.load('data/Y_v3(augmented-50).npy') 

#num_classes_y = 199
min_idx_y = 71
max_idx_y = 150
num_classes_y = max_idx_y - min_idx_y + 1

train_y = train_y - min_idx_y


# In[18]:


from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AvgPool1D, AvgPool2D, Reshape,
    Input, Activation, BatchNormalization, Dense, Add, Lambda, Dropout, LayerNormalization)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping

import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import elegy
import jax.numpy as jnp
import jax


# Let's define the newtork arquitecture. The simplified NN structure looks like this:
# 
# <img src="images/model_structure.png" style="width:680px;height:200px;">
# 
# "So the first block of convolutions learns to work with defense-offense pairs of players, using geometric features relative to rusher. The combination of multiple layers and activations before pooling was important to capture the trends properly. The second block of convolutions learns the necessary information per defense player before the aggregation. And the third block simply consists of dense layers and the usual things around them. 3 out of 5 input vectors do not depend on the offense player, hence they are constant across “off” dimension of the tensor." [1]

# In[27]:


from elegy.nn.transformers import Transformer, TransformerEncoder, TransformerEncoderLayer
from jax.experimental import optix

class Module(elegy.Module):
    
    def __init__(self, num_classes_y, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_y = num_classes_y

    def call(self, X):
        
        X = elegy.nn.Linear(64)(X)
        X = jax.nn.relu(X)
        X = elegy.nn.Linear(64)(X)
        
        X = TransformerEncoder(
            encoder_layer=lambda: TransformerEncoderLayer(
                head_size=16,
                num_heads=10,
                output_size=64,
                dropout=0.0,
                activation=jax.nn.relu,
            ),
            num_layers=3,
            norm=lambda: elegy.nn.LayerNormalization(),
        )(X)
        
        X = X[:, 0]
        
        elegy.add_summary("get_first", X)
        
        X = elegy.nn.Linear(96)(X)
        X = jax.nn.relu(X)
        X = elegy.nn.LayerNormalization()(X)

        X = elegy.nn.Linear(256)(X)
        X = jax.nn.relu(X)
        X = elegy.nn.LayerNormalization()(X)
#         X = elegy.nn.Dropout(0.3)(X)
        
        X = elegy.nn.Linear(self.num_classes_y)(X)
        X = jax.nn.softmax(X)

        return X
   

class CRPS(elegy.losses.MeanSquaredError):
    def call(self, y_true, y_pred):
        print(y_true[0], num_classes_y)
        y_true = jax.nn.one_hot(y_true, num_classes_y)
        y_true = jnp.cumsum(y_true, axis=1)
        
        y_pred = jnp.cumsum(y_pred, axis = 1)
        
        print(y_true[0])
        
        return super().call(y_true, y_pred)
    
model = elegy.Model(
    module=Module(num_classes_y),
    loss=CRPS(),
    optimizer=optix.adam(1e-4),
)

model.summary(train_x[:64], depth=1)


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
     train_x, train_y, test_size=0.33, random_state=42) 

model.run_eagerly = False
model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=100,
    steps_per_epoch=500,
    validation_data=(X_test, y_test),
    validation_steps=10,
)


# In[23]:


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_valid, y_valid = self.data[0], self.data[1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        logs['val_CRPS'] = val_s
        
        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


# In[26]:





# In[24]:


get_ipython().run_cell_magic('time', '', '\nmodels = []\nkf = KFold(n_splits=8, shuffle=True, random_state=42)\nscore = []\n\nfor i, (tdx, vdx) in enumerate(kf.split(train_x, train_y)):\n    print(f\'Fold : {i}\')\n    X_train, X_val = train_x[tdx], train_x[vdx],\n    y_train, y_val = train_y.iloc[tdx][\'YardIndexClipped\'].values, train_y.iloc[vdx][\'YardIndexClipped\'].values\n    season_val = df_season.iloc[vdx][\'Season\'].values\n\n    y_train_values = np.zeros((len(y_train), num_classes_y), np.int32)\n    for irow, row in enumerate(y_train):\n        y_train_values[(irow, row - min_idx_y)] = 1\n        \n    y_val_values = np.zeros((len(y_val), num_classes_y), np.int32)\n    for irow, row in enumerate(y_val - min_idx_y):\n        y_val_values[(irow, row)] = 1\n\n    val_idx = np.where(season_val!=2017)\n    \n    X_val = X_val[val_idx]\n    y_val_values = y_val_values[val_idx]\n\n    y_train_values = y_train_values.astype(\'float32\')\n    y_val_values = y_val_values.astype(\'float32\')\n    \n    model = get_conv_net(num_classes_y)\n\n    es = EarlyStopping(monitor=\'val_CRPS\',\n                        mode=\'min\',\n                        restore_best_weights=True,\n                        verbose=0,\n                        patience=10)\n    \n    es.set_model(model)\n    metric = Metric(model, [es], [X_val, y_val_values])\n\n    lr_i = 1e-3\n    lr_f = 5e-4\n    n_epochs = 50 \n\n    decay = (1-lr_f/lr_i)/((lr_f/lr_i)* n_epochs - 1)  #Time-based decay formula\n    alpha = (lr_i*(1+decay))\n    \n    opt = Adam(learning_rate=1e-3)\n    model.compile(loss=crps,\n                  optimizer=opt)\n    \n    model.fit(X_train,\n              y_train_values, \n              epochs=n_epochs,\n              batch_size=64,\n              verbose=0,\n              callbacks=[metric],\n              validation_data=(X_val, y_val_values))\n\n    val_crps_score = min(model.history.history[\'val_CRPS\'])\n    print("Val loss: {}".format(val_crps_score))\n    \n    score.append(val_crps_score)\n\n    models.append(model)\n    \nprint(np.mean(score))')


# In[25]:


print("The mean validation loss is {}".format(np.mean(score)))


# ## Conclusions

# With this elegant solution, The ZOO won the NFL Big Data Bowl 2020 (Kaggle competition). We submitted this code in kaggle and the score obtained was 0.011911 (2nd position in Leaderboard) [4]. A possible reason for the difference between our score and the winning score (0.011658) is that we do not implement TTA in our predictions. Moreover, the number of trainable parameters in our network structure differs (145,584) from the number that they report in [1] (145,329).
# 
# For further improvements, I would suggest trying to add features related to pitch control which was the VIP hint of the competition [5].

# [1] https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400
# 
# [2] https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview
# 
# [3] https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
# 
# [4] https://www.kaggle.com/jccampos/nfl-2020-winner-solution-the-zoo
# 
# [5] http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf

# In[ ]:




