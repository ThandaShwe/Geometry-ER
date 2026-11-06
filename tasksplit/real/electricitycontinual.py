import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class ElectricityLoadDataset:

    def __init__(
            self,
            file_path,
            target_column=None,
            task_type="monthly",
            num_tasks=5):        

        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.num_tasks = num_tasks

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()    

    def load_dataframe(self):        

        df = pd.read_csv(
            self.file_path,
            sep=";",
            decimal=","
        )        

        datetime_col = df.columns[0]

        df[datetime_col] = pd.to_datetime(
            df[datetime_col]
        )

        df = df.rename(
            columns={datetime_col: "datetime"}
        )        

        df = df.sort_values("datetime")        

        meter_columns = [
            c for c in df.columns
            if c != "datetime"
        ]

        if self.target_column is None:

            self.target_column = meter_columns[0]        

        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["weekday"] = df["datetime"].dt.weekday

        
        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )        

        df["lag_1"] = df[self.target_column].shift(1)
        df["lag_24"] = df[self.target_column].shift(24)

        # rolling mean
        df["rolling_mean_24"] = (
            df[self.target_column]
            .rolling(window=24)
            .mean()
        )        

        df = df.dropna()

        return df

    # Prepare Features    

    def prepare_data(self):

        df = self.load_dataframe()        
        y = df[self.target_column].values.reshape(-1, 1)        

        feature_cols = [
            c for c in df.columns
            if c not in [
                "datetime",
                self.target_column
            ]
        ]

        X = df[feature_cols].values        

        X = self.x_scaler.fit_transform(X)

        y = self.y_scaler.fit_transform(y)

        return (
            X.astype(np.float32),
            y.astype(np.float32),
            df
        )
    
    def create_tasks(self):

        X, y, df = self.prepare_data()
        tasks = []       

        if self.task_type == "monthly":
            months = sorted(
                df["month"].unique()
            )

            for month in months:

                indices = (
                    df["month"] == month
                )

                x_task = X[indices]
                y_task = y[indices]                

                split_idx = int(
                    0.8 * len(x_task)
                )

                x_train = x_task[:split_idx]
                y_train = y_task[:split_idx]

                x_test = x_task[split_idx:]
                y_test = y_task[split_idx:]

                tasks.append(
                    (
                        f"Month_{month}",
                        x_train,
                        y_train,
                        x_test,
                        y_test
                    )
                )

        # Sequential Tasks       

        elif self.task_type == "sequential":
            total_samples = len(X)
            chunk_size = (
                total_samples // self.num_tasks
            )
            for task_id in range(
                    self.num_tasks):

                start = task_id * chunk_size

                if task_id == self.num_tasks - 1:

                    end = total_samples

                else:

                    end = (
                        start + chunk_size
                    )

                x_task = X[start:end]
                y_task = y[start:end]

                # --------------------------------------------
                # Train/Test Split
                # --------------------------------------------

                split_idx = int(
                    0.8 * len(x_task)
                )

                x_train = x_task[:split_idx]
                y_train = y_task[:split_idx]

                x_test = x_task[split_idx:]
                y_test = y_task[split_idx:]

                tasks.append(
                    (
                        f"Task_{task_id+1}",
                        x_train,
                        y_train,
                        x_test,
                        y_test
                    )
                )

        return tasks