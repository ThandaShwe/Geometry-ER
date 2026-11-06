import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ExchangeRateDataset:

    def __init__(
        self,
        csv_path,
        seq_len=32,
        pred_len=8,
        task_type="yearly",
        normalize=True
    ):

        self.csv_path = csv_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.task_type = task_type
        self.normalize = normalize
        self.scaler = StandardScaler()
    

    def load_dataframe(self):

        df = pd.read_csv(self.csv_path)
        print("Original Shape:", df.shape)     

        first_col = df.columns[0]
        df = df[
            ~df[first_col].astype(str).str.contains(
                "Unit:|Currency:|Date",
                na=False
            )
        ]
        
        df.rename(columns={first_col: "date"}, inplace=True)        
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )
        df = df.dropna(subset=["date"])     

        df = df.replace("ND", np.nan)
        df = df.replace("NA", np.nan)
        df = df.replace("", np.nan)        

        feature_cols = [c for c in df.columns if c != "date"]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.ffill().bfill()
        print("After Cleaning:", df.shape)
        return df

    

    def create_tasks(
        self,
        sequence_length=32,
        pred_horizon=8,
        min_task_size=100,
        test_size=0.2
    ):

        df = self.load_dataframe()      

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter        

        feature_cols = [
            c for c in df.columns
            if c not in ["date"]
        ]

        target_col = feature_cols[0]

        print("Features:", feature_cols)
        print("Target:", target_col)        

        data_cols = feature_cols

        if self.normalize:
            df[data_cols] = self.scaler.fit_transform(
                df[data_cols]
            )        

        if self.task_type == "monthly":

            df["task_id"] = (
                df["year"].astype(str)
                + "_"
                + df["month"].astype(str).str.zfill(2)
            )

        elif self.task_type == "quarterly":

            df["task_id"] = (
                df["year"].astype(str)
                + "_Q"
                + df["quarter"].astype(str)
            )

        elif self.task_type == "yearly":

            df["task_id"] = df["year"].astype(str)

        else:
            raise ValueError("Unknown task_type")        

        unique_tasks = sorted(df["task_id"].unique())
        TASKS = []
        task_counter = 0
        for task_name in unique_tasks:

            task_df = df[df["task_id"] == task_name]

            if len(task_df) < min_task_size:
                continue

            data = task_df[data_cols].values.astype(np.float32)

            X_list = []
            y_list = []
            
            for i in range(
                len(data) - sequence_length - pred_horizon
            ):

                x = data[i:i + sequence_length]

                y = data[
                    i + sequence_length:
                    i + sequence_length + pred_horizon,
                    0
                ]

                X_list.append(x)
                y_list.append(y)

            if len(X_list) == 0:
                continue

            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)         

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=False
            )

            TASKS.append(
                (
                    task_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
            )

            print(f"\nTask {task_counter}: {task_name}")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("X_test :", X_test.shape)
            print("y_test :", y_test.shape)

            task_counter += 1

        print("\nTotal Tasks:", len(TASKS))

        return TASKS