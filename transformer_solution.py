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


import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
import typer
from jax.experimental import optix
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import elegy
import utils
from elegy.nn.transformers import (
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def main(cache: bool = False):

    if not cache or not Path("data/X_v4.npy").exists():

        train = pd.read_csv("data/train.csv", dtype={"WindSpeed": "object"})
        train["Yards"] -= train["Yards"].min()
        train_x, train_y = preprocess_data(train)

        print(train_x.shape, train_y.shape)

        np.save("data/X_v4.npy", train_x)
        np.save("data/Y_v4.npy", train_y)
    else:

        train_x: np.ndarray = np.load("data/X_v4.npy")
        train_y: np.ndarray = np.load("data/Y_v4.npy")

    num_classes_y = train_y.max() + 1

    plt.hist(train_y, bins=np.unique(train_y))
    plt.show()

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

            # X = elegy.nn.Linear(96)(X)
            # X = jax.nn.relu(X)
            # X = elegy.nn.LayerNormalization()(X)

            # X = elegy.nn.Linear(256)(X)
            # X = jax.nn.relu(X)
            # X = elegy.nn.LayerNormalization()(X)
            # X = elegy.nn.Dropout(0.3)(X)

            X = elegy.nn.Linear(self.num_classes_y)(X)
            X = jax.nn.softmax(X)

            return X

    class CRPS(elegy.metrics.Mean):
        def call(self, y_true, y_pred):
            y_true = jax.nn.one_hot(y_true, num_classes_y)
            y_true = jnp.cumsum(y_true, axis=1)
            y_pred = jnp.cumsum(y_pred, axis=1)

            values = jnp.sum(jnp.square(y_true - y_pred), axis=1)

            return super().call(values)

    model = elegy.Model(
        module=Module(num_classes_y),
        loss=elegy.losses.SparseCategoricalCrossentropy(),
        metrics=CRPS(),
        optimizer=optix.adam(1e-4),
    )

    model.summary(train_x[:64], depth=1)

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=0.33, random_state=42
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=64,
        epochs=2,
        steps_per_epoch=500,
        validation_data=(X_test, y_test),
        validation_steps=10,
    )

    preds = model.predict(X_test[:10])

    for pred in preds:
        plt.plot(pred)
        plt.show()


def preprocess_data(train):
    def split_play_and_player_cols(df, predicting=False):
        df["IsRusher"] = df["NflId"] == df["NflIdRusher"]

        df["PlayId"] = df["PlayId"].astype(str)

        # We must assume here that the first 22 rows correspond to the same player:
        player_cols = [
            "PlayId",  # This is the link between them
            "Season",
            "Team",
            "X",
            "Y",
            "S",
            "Dis",
            "Dir",
            "NflId",
            "IsRusher",
            "Yards",
        ]

        df_players = df[player_cols]

        play_cols = [
            "PlayId",
            "Season",
            "PossessionTeam",
            "HomeTeamAbbr",
            "VisitorTeamAbbr",
            "PlayDirection",
            "FieldPosition",
            "YardLine",
        ]

        if not predicting:
            play_cols.append("Yards")

        df_play = df[play_cols].copy()

        # Get first
        df_play = df_play.groupby("PlayId").first().reset_index()

        print("rows/plays in df: ", len(df_play))
        assert (
            df_play.PlayId.nunique() == df.PlayId.nunique()
        ), "Play/player split failed?"  # Boom

        return df_play, df_players

    play_ids = train["PlayId"].unique()

    df_play, df_players = split_play_and_player_cols(train)

    def process_team_abbr(df):

        # These are only problems:
        map_abbr = {"ARI": "ARZ", "BAL": "BLT", "CLE": "CLV", "HOU": "HST"}
        for abb in train["PossessionTeam"].unique():
            map_abbr[abb] = abb

        df["PossessionTeam"] = df["PossessionTeam"].map(map_abbr)
        df["HomeTeamAbbr"] = df["HomeTeamAbbr"].map(map_abbr)
        df["VisitorTeamAbbr"] = df["VisitorTeamAbbr"].map(map_abbr)
        df["HomePossession"] = df["PossessionTeam"] == df["HomeTeamAbbr"]

        return

    process_team_abbr(df_play)

    def process_play_direction(df):
        df["IsPlayLeftToRight"] = df["PlayDirection"].apply(
            lambda val: True if val.strip() == "right" else False
        )
        return

    process_play_direction(df_play)

    # We compute how many yards are left to the end-zone.

    def process_yard_til_end_zone(df):
        def convert_to_yardline100(row):
            return (
                (100 - row["YardLine"])
                if (row["PossessionTeam"] == row["FieldPosition"])
                else row["YardLine"]
            )

        df["Yardline100"] = df.apply(convert_to_yardline100, axis=1)
        return

    process_yard_til_end_zone(df_play)

    # Now, we add the computed features to df_players

    df_players = df_players.merge(
        df_play[
            [
                "PlayId",
                "PossessionTeam",
                "HomeTeamAbbr",
                "PlayDirection",
                "Yardline100",
            ]
        ],
        how="left",
        on="PlayId",
    )

    df_players.loc[df_players.Season == 2017].plot.scatter(
        x="Dis", y="S", title="Season 2017", grid=True
    )

    df_players.loc[df_players.Season == 2018].plot.scatter(
        x="Dis", y="S", title="Season 2018", grid=True
    )

    X = df_players.loc[df_players.Season == 2018]["Dis"]
    y = df_players.loc[df_players.Season == 2018]["S"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    model.summary()

    df_players.loc[df_players.Season == 2017, "S"] = (
        10 * df_players.loc[df_players.Season == 2017, "Dis"]
    )

    def standarize_direction(df):
        # adjusted the data to always be from left to right
        df["HomePossesion"] = df["PossessionTeam"] == df["HomeTeamAbbr"]

        df["Dir_rad"] = np.mod(90 - df.Dir, 360) * math.pi / 180.0

        df["ToLeft"] = df.PlayDirection == "left"
        df["TeamOnOffense"] = "home"
        df.loc[df.PossessionTeam != df.HomeTeamAbbr, "TeamOnOffense"] = "away"
        df["IsOnOffense"] = df.Team == df.TeamOnOffense  # Is player on offense?
        df["X_std"] = df.X
        df.loc[df.ToLeft, "X_std"] = 120 - df.loc[df.ToLeft, "X"]
        df["Y_std"] = df.Y
        df.loc[df.ToLeft, "Y_std"] = 160 / 3 - df.loc[df.ToLeft, "Y"]
        df["Dir_std"] = df.Dir_rad
        df.loc[df.ToLeft, "Dir_std"] = np.mod(
            np.pi + df.loc[df.ToLeft, "Dir_rad"], 2 * np.pi
        )

        # Replace Null in Dir_rad
        df.loc[(df.IsOnOffense) & df["Dir_std"].isna(), "Dir_std"] = 0.0
        df.loc[~(df.IsOnOffense) & df["Dir_std"].isna(), "Dir_std"] = np.pi

    standarize_direction(df_players)

    for play_id in ["20170910001102", "20170910000081"]:
        utils.show_play(play_id, df_players)

    for play_id in ["20170910001102", "20170910000081"]:
        utils.show_play_std(play_id, df_players)

    def data_augmentation(df, sample_ids):
        df_sample = df.loc[df.PlayId.isin(sample_ids)].copy()
        df_sample["Y_std"] = 160 / 3 - df_sample["Y_std"]
        df_sample["Dir_std"] = df_sample["Dir_std"].apply(lambda x: 2 * np.pi - x)
        df_sample["PlayId"] = df_sample["PlayId"].apply(lambda x: x + "_aug")
        return df_sample

    def process_tracking_data(df):
        # More feature engineering for all:
        df["Sx"] = df["S"] * df["Dir_std"].apply(math.cos)
        df["Sy"] = df["S"] * df["Dir_std"].apply(math.sin)

        # ball carrier position
        rushers = df[df["IsRusher"]].copy()
        rushers.set_index("PlayId", inplace=True, drop=True)
        playId_rusher_map = rushers[["X_std", "Y_std", "Sx", "Sy"]].to_dict(
            orient="index"
        )
        rusher_x = df["PlayId"].apply(lambda val: playId_rusher_map[val]["X_std"])
        rusher_y = df["PlayId"].apply(lambda val: playId_rusher_map[val]["Y_std"])
        rusher_Sx = df["PlayId"].apply(lambda val: playId_rusher_map[val]["Sx"])
        rusher_Sy = df["PlayId"].apply(lambda val: playId_rusher_map[val]["Sy"])

        # Calculate differences between the rusher and the players:
        df["player_minus_rusher_x"] = rusher_x - df["X_std"]
        df["player_minus_rusher_y"] = rusher_y - df["Y_std"]

        # Velocity parallel to direction of rusher:
        df["player_minus_rusher_Sx"] = rusher_Sx - df["Sx"]
        df["player_minus_rusher_Sy"] = rusher_Sy - df["Sy"]

        return

    sample_ids = np.random.choice(
        df_play.PlayId.unique(), int(0.5 * len(df_play.PlayId.unique()))
    )

    df_players_aug = data_augmentation(df_players, sample_ids)
    df_players = pd.concat([df_players, df_players_aug])
    df_players.reset_index()

    df_play_aug = df_play.loc[df_play.PlayId.isin(sample_ids)].copy()
    df_play_aug["PlayId"] = df_play_aug["PlayId"].apply(lambda x: x + "_aug")
    df_play = pd.concat([df_play, df_play_aug])
    df_play.reset_index()

    # This is necessary to maintain the order when in the next cell we use groupby
    df_players.sort_values(by=["PlayId"], inplace=True)
    df_play.sort_values(by=["PlayId"], inplace=True)

    process_tracking_data(df_players)

    tracking_level_features = [
        "PlayId",
        "IsOnOffense",
        "X_std",
        "Y_std",
        "Sx",
        "Sy",
        "player_minus_rusher_x",
        "player_minus_rusher_y",
        "player_minus_rusher_Sx",
        "player_minus_rusher_Sy",
        "IsRusher",
        "Yards",
    ]

    df_all_feats = df_players[tracking_level_features]

    print("Any null values: ", df_all_feats.isnull().sum().sum())

    df_all_feats.columns

    Xs = []
    Ys = []

    df_all_feats["Yards"].plot.hist()
    plt.show()

    groups = df_all_feats[
        ["PlayId", "Yards", "X_std", "Y_std", "Sx", "Sy", "IsRusher"]
    ].groupby("PlayId")

    for play_id, df in tqdm(groups):
        df = df.sort_values("IsRusher", ascending=False)
        Xs.append(
            df[["X_std", "Y_std", "Sx", "Sy", "IsRusher"]].to_numpy().astype(np.float32)
        )

        Ys.append(df["Yards"].iloc[0])

    train_x: np.ndarray = np.stack(Xs, axis=0)
    train_y: np.ndarray = np.stack(Ys, axis=0)

    return train_x, train_y


if __name__ == "__main__":
    typer.run(main)
