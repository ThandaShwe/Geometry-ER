import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class BikeSharingDataset:

    def __init__(
            self,
            file_path,
            target_col="count",
            task_type="season"):

        self.file_path = file_path
        self.target_col = target_col
        self.task_type = task_type      

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

   
    def load_dataframe(self):

        df = pd.read_csv(self.file_path)

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        df = df.sort_values("datetime")        

        df["hour"] = df["datetime"].dt.hour

        df["day"] = df["datetime"].dt.day

        df["month"] = df["datetime"].dt.month

        df["weekday"] = df["datetime"].dt.weekday

        return df    

    def create_task_dataframes(self):

        df = self.load_dataframe()

        tasks = []
        
        if self.task_type == "season":

            season_map = {
                1: "Winter",
                2: "Spring",
                3: "Summer",
                4: "Fall"
            }

            for season_id in [1, 2, 3, 4]:

                task_df = df[
                    df["season"] == season_id
                ].copy()

                tasks.append(
                    (
                        season_map[season_id],
                        task_df
                    )
                )
        
        elif self.task_type == "month":

            for month in range(1, 13):

                task_df = df[
                    df["month"] == month
                ].copy()

                tasks.append(
                    (
                        f"Month_{month}",
                        task_df
                    )
                )

        else:

            raise ValueError(
                f"Unknown task type: {self.task_type}"
            )

        return tasks
    

    def prepare_data(self):

        df = self.load_dataframe()        

        if self.task_type == "season":

            task_labels = df["season"].values

        else:

            task_labels = df["month"].values        

        df = df.drop(columns=["datetime"])       

        y_raw = df[
            self.target_col
        ].values.reshape(-1, 1)        

        feature_cols = [
            col for col in df.columns
            if col != self.target_col
        ]

        X_raw = df[feature_cols].values        

        X_scaled = self.x_scaler.fit_transform(
            X_raw
        )

        y_scaled = self.y_scaler.fit_transform(
            y_raw
        )

        return (
            X_scaled.astype(np.float32),
            y_scaled.astype(np.float32),
            task_labels
        )
    
    def create_tasks(self):

        X, y, task_labels = self.prepare_data()

        unique_tasks = sorted(
            np.unique(task_labels)
        )

        tasks = []

        for task_id in unique_tasks:

            indices = np.where(
                task_labels == task_id
            )[0]

            X_task = X[indices]

            y_task = y[indices]            

            split_idx = int(
                0.8 * len(X_task)
            )

            x_train = X_task[:split_idx]

            y_train = y_task[:split_idx]

            x_test = X_task[split_idx:]

            y_test = y_task[split_idx:]            

            if self.task_type == "season":

                season_map = {
                    1: "Winter",
                    2: "Spring",
                    3: "Summer",
                    4: "Fall"
                }

                task_name = season_map[task_id]

            else:

                task_name = f"Month_{task_id}"

            tasks.append(
                (
                    task_name,
                    x_train,
                    y_train,
                    x_test,
                    y_test
                )
            )

        return tasks