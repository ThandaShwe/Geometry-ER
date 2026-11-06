import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class ETTH1Dataset:  

    def __init__(
            self,
            file_path,
            target_col="OT",
            seq_len=24):

        self.file_path = file_path
        self.target_col = target_col

        # Number of previous time steps
        self.seq_len = seq_len
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()    

    def load_dataframe(self):

        df = pd.read_csv(self.file_path)      

        df["date"] = pd.to_datetime(df["date"])       

        df = df.sort_values("date")

        return df
    
    def add_time_features(self, df):

        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["weekday"] = df["date"].dt.weekday

        return df    

    def prepare_data(self):

        df = self.load_dataframe()

        df = self.add_time_features(df)       

        months = df["month"].values     
       
        feature_cols = [
            col for col in df.columns
            if col != "date"
        ]

        X_raw = df[feature_cols].values

        y_raw = df[self.target_col].values.reshape(-1, 1)        

        X_scaled = self.x_scaler.fit_transform(X_raw)

        y_scaled = self.y_scaler.fit_transform(y_raw)        

        X_seq = []
        y_seq = []
        month_seq = []

        for i in range(self.seq_len, len(df)):

            # Previous seq_len timesteps
            x_window = X_scaled[
                i - self.seq_len:i
            ]

            # Predict current timestep
            y_target = y_scaled[i]

            X_seq.append(x_window.flatten())

            y_seq.append(y_target)

            month_seq.append(months[i])

        X_seq = np.array(X_seq, dtype=np.float32)

        y_seq = np.array(y_seq, dtype=np.float32)

        month_seq = np.array(month_seq)

        return X_seq, y_seq, month_seq

   
    def create_tasks(self):

        X, y, months = self.prepare_data()

        unique_months = sorted(
            np.unique(months)
        )

        tasks = []
        for month in unique_months:

            indices = np.where(
                months == month
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

            task_name = f"Month_{month}"
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