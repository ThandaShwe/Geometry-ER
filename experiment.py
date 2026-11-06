import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from reserviorbuffer import ReservoirBuffer
from geometrybuffer import GeometryBuffer
from clsamplingbuffer import CLSamplingBuffer
from data.real.electricitycontinual import ElectricityLoadDataset
from data.real.ettcontinual import ETTH1Dataset
from data.real.METAcontinual import METRLADataset
from data.real.PEMScontinual import PEMSBAYDataset
from data.real.ettm2continual import ETTM2Dataset
from data.real.JenaClimateContinual import JenaClimateDataset
from data.real.exchangeRateContinual import ExchangeRateDataset
from data.real.bikesharingcontinual import BikeSharingDataset
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from resource_monitor import ResourceMonitor


SEED=42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Visualizations

def visualize_distribution_shift(TASKS):
    num_tasks = len(TASKS)
    matrix = np.zeros((num_tasks, num_tasks))

    for i in range(num_tasks):

        _, xi_train, _, _, _ = TASKS[i]
        xi = xi_train.reshape(xi_train.shape[0],-1)

        for j in range(num_tasks):

            _, xj_train, _, _, _ = TASKS[j]
            xj = xj_train.reshape(xj_train.shape[0], -1)
            dist = wasserstein_distance(xi.flatten(),xj.flatten())
            matrix[i, j] = dist    

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,cmap="viridis",annot=True)
    plt.title("Distribution Shift Between Tasks")
    plt.xlabel("Task")
    plt.ylabel("Task")
    plt.show()

def plot_sequential_drift(TASKS):

    drifts = []

    for i in range(len(TASKS)-1):
        _, x1_train, _, _, _ = TASKS[i]
        _, x2_train, _, _, _ = TASKS[i+1]
        x1 = x1_train.reshape(-1)
        x2 = x2_train.reshape(-1)
        drift = wasserstein_distance(x1, x2)
        drifts.append(drift)

    plt.figure(figsize=(12,5))
    plt.plot(drifts, marker='o')
    plt.xlabel("Task Transition")
    plt.ylabel("Distribution Distance")
    plt.title("Sequential Distribution Drift")
    plt.grid(True)
    plt.show() 


def visualize_task_embeddings(TASKS):

    task_vectors = []
    for task in TASKS:
        _, x_train, _, _, _ = task
        vec = np.mean(x_train, axis=0).flatten()
        task_vectors.append(vec)

    task_vectors = np.array(task_vectors)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(task_vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1])
    for i in range(len(reduced)):
        plt.text(
            reduced[i,0],
            reduced[i,1],
            str(i)
        )

    plt.title("Task Distribution Embedding (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_task_evolution_pca(TASKS):

    task_reps = []
    for task in TASKS:
        _, x_train, _, _, _ = task
        x_flat = x_train.reshape(-1,x_train.shape[-1])
        rep = np.mean(x_flat, axis=0)
        task_reps.append(rep)

    task_reps = np.array(task_reps)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(task_reps)
    plt.figure(figsize=(8, 8))
    plt.plot(reduced[:, 0],reduced[:, 1],marker='o')

    for i in range(len(reduced)):
        plt.text(reduced[i, 0], reduced[i, 1],str(i))
    plt.title("Task Evolution in Feature Space")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

def compute_task_representations(TASKS):

    representations = []

    for task in TASKS:
        _, x_train, _, _, _ = task        
        x_flat = x_train.reshape(-1,x_train.shape[-1])        
        rep = np.mean(x_flat, axis=0)
        representations.append(rep)
    return np.array(representations)

def compute_consecutive_drift(task_reps):

    drifts = []
    for i in range(len(task_reps) - 1):
        d = euclidean(task_reps[i],task_reps[i + 1])
        drifts.append(d)

    return drifts

def plot_drift_trajectory(drifts):

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(drifts)), drifts,marker='o')
    plt.xlabel("Task Transition")
    plt.ylabel("Distribution Drift")
    plt.title("Distribution Shift Between Consecutive Tasks")
    plt.grid(True)
    plt.show()

# datasets loading

""" dataset = BikeSharingDataset(
    file_path="data/real/bikeshare.csv",  
    target_col="count"
) """

dataset = ElectricityLoadDataset(
    file_path="data/real/LD2011_2014.txt",
    target_column="MT_001",
    task_type="monthly",
    #num_tasks=5 
)

""" dataset = ETTH1Dataset(
    file_path="data/real/ETTh1.csv",
    target_col="OT",
    seq_len=24
) """

""" dataset = METRLADataset(
    file_path="data/real/METR-LA.csv",
    seq_len=12
) """

""" dataset = PEMSBAYDataset(
    file_path="data/real/PEMS-BAY.csv",
    seq_len=12
) """

""" dataset = ETTM2Dataset(
    file_path="data/real/ETTm2.csv",
    target_col="OT",
    seq_len=24
) """

""" dataset = JenaClimateDataset(
    file_path="data/real/jena_climate.csv",
    target_col="T (degC)",
    seq_len=24
) """  


TASKS = dataset.create_tasks(   
    #num_tasks=5
)
total_train_samples = sum(len(task[1]) for task in TASKS)
total_test_samples = sum(len(task[3]) for task in TASKS)
total_samples = total_train_samples + total_test_samples


print("DATASET INFORMATION")
print("=" * 60)
print(f"Number of tasks       : {len(TASKS)}")
print(f"Total training samples: {total_train_samples}")
print(f"Total test samples    : {total_test_samples}")
print(f"Total dataset samples : {total_samples}")

# Check characteristics of datasets

""" visualize_distribution_shift(TASKS)
plot_sequential_drift(TASKS)
visualize_task_embeddings(TASKS)

task_reps = compute_task_representations(TASKS)

drifts = compute_consecutive_drift(task_reps)

plot_drift_trajectory(drifts)
plot_task_evolution_pca(TASKS) """

# Model

class MLPRegressor(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.net(x)


# Training Part

def train_model(model,x_train,y_train,epochs=50,lr=0.001):

    model.train()

    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x_tensor)
        loss = criterion(preds,y_tensor)
        loss.backward()
        optimizer.step()
    return model


# Evaluation Part

def evaluate_model(model,x_test,y_test):

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_test,dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy()

    mae = mean_absolute_error(y_test,preds)
    return mae

# Calculate metrics

def compute_final_avg_mae(results):

    final_row = results[-1]
    valid = final_row[final_row > 0]
    return np.mean(valid)

def compute_final_avg_forgetting(results):

    num_tasks = results.shape[0]
    forgetting_scores = []
    for task in range(num_tasks - 1):
        best_previous = np.min(results[:num_tasks-1, task])
        final_performance = results[-1, task]
        forgetting = (final_performance- best_previous)
        forgetting_scores.append(forgetting)

    return np.mean(forgetting_scores)

# check execution time
def run_timed_experiment(method, **kwargs):
    start_time = time.perf_counter()

    results = continual_learning_experiment(
        method=method,
        **kwargs
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"{method}: {elapsed_time:.2f} seconds "
          f"({elapsed_time / 60:.2f} minutes)")

    return results, elapsed_time

# main experiment loop

def continual_learning_experiment(
        method="none",
        memory_size=500,
        replay_batch_size=256):

    print("=" * 60)
    print(f"METHOD: {method.upper()}")
    print("=" * 60)    
    
    input_dim = TASKS[0][1].shape[1]  

    model = MLPRegressor(input_dim=input_dim).to(device)

    # Buffer Selection
    
    replay_buffer = None

    if method == "reservoir":
        replay_buffer = ReservoirBuffer(max_size=memory_size)    

    elif method == "geometry":
        replay_buffer = GeometryBuffer(max_size=memory_size)    
    
    elif method == "clsampling":
            replay_buffer = CLSamplingBuffer(max_size=memory_size, input_dim=input_dim,device=device)       
    
   
    num_tasks = len(TASKS)

    results = np.zeros((num_tasks, num_tasks))

    all_task_data = []

    # Sequential Training

    for task_id, (task_name,x_train,y_train,x_test,y_test) in enumerate(TASKS):

        print("\n" + "-" * 50)
        print(f"Training on {task_name}")
        print("-" * 50)

        all_task_data.append((x_train,y_train,x_test,y_test))        

        train_x = x_train
        train_y = y_train        

        # Full replay        

        if method == "full":

            replay_x = []
            replay_y = []

            for old_data in all_task_data:

                replay_x.append(old_data[0])
                replay_y.append(old_data[1])

            train_x = np.concatenate(replay_x,axis=0)

            train_y = np.concatenate(replay_y,axis=0)

        # Replay methods        

        elif method in [
            "reservoir",            
            "geometry",                         
            "clsampling"]:

            if task_id > 0:

                if method in ["reservoir", "geometry"]:
                    replay_x, replay_y = replay_buffer.sample(replay_batch_size)                
                elif method == "clsampling":
                    replay_x, replay_y = replay_buffer.sample(replay_batch_size,model)         
                                                  

                if replay_x is not None and replay_y is not None:
                    train_x = np.concatenate([x_train, replay_x], axis=0)
                    train_y = np.concatenate([y_train, replay_y], axis=0)

        print(f"Task {task_id + 1} ({task_name})")
        print(f"  Original samples: {len(x_train)}")
        print(f"  Training samples: {len(train_x)}")
        print(f"  Training labels:  {len(train_y)}")
        
        if method != "none" and method != "full" and task_id > 0:
            print(f"  Replay samples:   {len(train_x) - len(x_train)}")
        
        model = train_model(model, train_x, train_y)

        # Buffer update

        if method in ["reservoir","geometry"]:         
            replay_buffer.add_batch(x_train,y_train,task_id=task_id)        

        elif method == "clsampling":
            replay_buffer.add_batch(model, x_train, y_train,task_id)
        
        
        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------
        for eval_task in range(task_id + 1):
            _, _, x_eval, y_eval = all_task_data[eval_task]
            mae = evaluate_model(model, x_eval, y_eval)
            results[task_id, eval_task] = mae     
           


    # Calculate metrics

    final_avg_mae = compute_final_avg_mae(results)
    final_avg_forgetting = compute_final_avg_forgetting(results)

    output_file = f"final_results.txt"

    with open(output_file, "a") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write(f"{method.upper()} FINAL RESULTS\n")
        f.write("=" * 60 + "\n")

        f.write(f"Final Average MAE: {final_avg_mae:.4f}\n")
        f.write(f"Final Average Forgetting: {final_avg_forgetting:.4f}\n")

        f.write("=" * 60 + "\n")

    print(f"Results saved to: {output_file}")
    return results

monitor = ResourceMonitor(interval=1.0)

monitor.start()

experiment_times = {}

set_seed(SEED)
results_none, experiment_times["None"] = run_timed_experiment(
    method="none" 
)

""" set_seed(SEED)
results_full, experiment_times["Full"] = run_timed_experiment(
    method="full"
) """

""" set_seed(SEED)
results_reservoir, experiment_times["Reservoir"] = run_timed_experiment(
    method="reservoir",
    memory_size=100000,
    replay_batch_size=10000
) """

""" set_seed(SEED)
results_geometry, experiment_times["Geometry"] = run_timed_experiment(
    method="geometry",
    memory_size=100000,
    replay_batch_size=10000
) """

""" set_seed(SEED)
results_clsampling, experiment_times["CLSampling"] = run_timed_experiment(
    method="clsampling",
    memory_size=30000,
    replay_batch_size=3000
) """ 

monitor.stop()
summary = monitor.save()
print(summary)

timing_df = pd.DataFrame([
    {
        "Method": method,
        "Time (seconds)": elapsed,
        "Time (minutes)": elapsed / 60
    }
    for method, elapsed in experiment_times.items()
])

timing_df = timing_df.sort_values("Time (seconds)")

print(timing_df.round(2).to_string(index=False))
# ============================================================
# Plotting
# ============================================================

def plot_results(results, title):

    plt.figure(figsize=(8, 6))
    plt.imshow(results,cmap="viridis",interpolation="nearest")
    plt.colorbar(label="MSE")
    plt.xlabel("Evaluation Task")
    plt.ylabel("Training Task")
    plt.title(title)
    plt.xticks(range(len(TASKS)),[f"T{i+1}" for i in range(len(TASKS))])

    plt.yticks(range(len(TASKS)),[f"T{i+1}" for i in range(len(TASKS))])
    plt.show()

    """ plot_results(results_none, "No Replay")
    plot_results(results_full, "Full Replay")

    plot_results(results_reservoir,"Reservoir Replay")

    plot_results(results_uncertainty, "Uncertainty Replay")

    plot_results(results_geometry,"Geometry Replay")
    plot_results(results_clsampling, "CLsampling Replay") """
 