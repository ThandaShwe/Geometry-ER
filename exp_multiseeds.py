import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from reserviorbuffer import ReservoirBuffer
from uncertaintybuffer import UncertaintyBuffer
from geometrybuffer import GeometryBuffer
from clsamplingbuffer import CLSamplingBuffer
from prototype_replay_new import MemoryEfficientReplayRegressor
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

# ============================================================
# Reproducibility
# ============================================================

# The same seed list is used for every method so comparisons are
# performed under the same random-seed conditions.
SEEDS = [42, 43, 44, 45, 46]

def set_seed(seed):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA behavior where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Distribution Shift Heatmap
# ============================================================

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

    # --------------------------------------------
    # Plot
    # --------------------------------------------

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
# ============================================================
# Load Dataset
# ============================================================

""" dataset = BikeSharingDataset(
    file_path="data/real/bikeshare.csv",  
    target_col="count"
) """

""" dataset = ElectricityLoadDataset(
    file_path="data/real/LD2011_2014.txt",
    target_column="MT_001",
    task_type="monthly",
    #num_tasks=5 
) """

""" dataset = ETTH1Dataset(
    file_path="data/real/ETTh1.csv",
    target_col="OT",
    seq_len=24
) """

""" dataset = METRLADataset(
    file_path="data/real/METR-LA.csv",
    seq_len=12
) """

dataset = PEMSBAYDataset(
    file_path="data/real/PEMS-BAY.csv",
    seq_len=12
)

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
 
""" dataset = ExchangeRateDataset(
        csv_path="data/real/exchange_rates.csv",
        seq_len=32, 
        pred_len=8,
        task_type="yearly",  # monthly / quarterly / yearly        
        normalize=True
    ) """

TASKS = dataset.create_tasks(   
    #num_tasks=5
)
total_train_samples = sum(len(task[1]) for task in TASKS)
total_test_samples = sum(len(task[3]) for task in TASKS)
total_samples = total_train_samples + total_test_samples

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Number of tasks       : {len(TASKS)}")
print(f"Total training samples: {total_train_samples}")
print(f"Total test samples    : {total_test_samples}")
print(f"Total dataset samples : {total_samples}")
print("=" * 60)
# ============================================================
# Visualizations
# ============================================================

""" visualize_distribution_shift(TASKS)
plot_sequential_drift(TASKS)
visualize_task_embeddings(TASKS)

task_reps = compute_task_representations(TASKS)

drifts = compute_consecutive_drift(task_reps)

plot_drift_trajectory(drifts)
plot_task_evolution_pca(TASKS) """

# ============================================================
# Model
# ============================================================

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


# ============================================================
# Training
# ============================================================

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

# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, x_test, y_test):

    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(
            x_test,
            dtype=torch.float32
        ).to(device)

        preds = model(x_tensor).cpu().numpy()

    mae = mean_absolute_error(y_test, preds)

    return mae

# ============================================================
# Metrics
# ============================================================

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
# ============================================================
# Metrics
# ============================================================

""" def compute_final_avg_mae(results):

    final_row = results[-1]

    return np.nanmean(final_row)


def compute_final_avg_forgetting(results):

    num_tasks = results.shape[0]
    forgetting_scores = []

    # Calculate forgetting for all tasks except the final task.
    for task in range(num_tasks - 1):

        # Performance history starting from when the task
        # was first learned.
        task_history = results[task:, task]

        # Find the best performance before the final evaluation.
        previous_performance = task_history[:-1]

        previous_performance = previous_performance[
            ~np.isnan(previous_performance)
        ]

        if len(previous_performance) == 0:
            continue

        # MAE is a lower-is-better metric.
        best_previous = np.min(previous_performance)

        final_performance = results[-1, task]

        forgetting = (
            final_performance - best_previous
        )

        forgetting_scores.append(forgetting)

    if len(forgetting_scores) == 0:
        return np.nan

    return np.mean(forgetting_scores) """

# ============================================================
# Continual Learning Experiment
# ============================================================

def continual_learning_experiment(
        method="none",
        memory_size=500,
        replay_batch_size=256):

    print("=" * 60)
    print(f"METHOD: {method.upper()}")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Input dimension. change here for exchangeRatedataset
    # --------------------------------------------------------

    input_dim = TASKS[0][1].shape[1]
    #input_dim = TASKS[0][1].shape[2]

    model = MLPRegressor(input_dim=input_dim).to(device)

    # --------------------------------------------------------
    # Replay buffer
    # --------------------------------------------------------
    
    replay_buffer = None

    if method == "reservoir":
        replay_buffer = ReservoirBuffer(max_size=memory_size)

    elif method == "uncertainty":
        replay_buffer = UncertaintyBuffer(max_size=memory_size, mc_passes=10)

    elif method == "geometry":
        replay_buffer = GeometryBuffer(max_size=memory_size)    
    
    elif method == "clsampling":
            replay_buffer = CLSamplingBuffer(max_size=memory_size, input_dim=input_dim,device=device)
    elif method == "memory_efficient":
        mer_model = MemoryEfficientReplayRegressor(input_dim=input_dim,device=device)     

    
    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    num_tasks = len(TASKS)

    results = np.zeros((num_tasks, num_tasks))
    """ results = np.full(
    (num_tasks, num_tasks),
    np.nan
    ) """

    all_task_data = []

    # ========================================================
    # Sequential Learning
    # ========================================================

    for task_id, (task_name,x_train,y_train,x_test,y_test) in enumerate(TASKS):

        print("\n" + "-" * 50)
        print(f"Training on {task_name}")
        print("-" * 50)

        all_task_data.append((x_train,y_train,x_test,y_test))

        # ----------------------------------------------------
        # Default training
        # ----------------------------------------------------

        train_x = x_train
        train_y = y_train        

        # ----------------------------------------------------
        # Full replay
        # ----------------------------------------------------

        if method == "full":

            replay_x = []
            replay_y = []

            for old_data in all_task_data:

                replay_x.append(old_data[0])
                replay_y.append(old_data[1])

            train_x = np.concatenate(replay_x,axis=0)

            train_y = np.concatenate(replay_y,axis=0)

        # ----------------------------------------------------
        # Replay methods
        # ----------------------------------------------------

        elif method in [
            "reservoir",
            "uncertainty",
            "geometry",                         
            "clsampling"]:

            if task_id > 0:

                if method in ["reservoir", "geometry"]:
                    replay_x, replay_y = replay_buffer.sample(replay_batch_size)

                elif method == "uncertainty":
                    replay_x, replay_y = replay_buffer.sample(replay_batch_size,model,device)
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
        
        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------
        # ----------------------------------------------------
        # Train (MER baseline)
        # ----------------------------------------------------
        if method == "memory_efficient":
                mer_model.observe(x_train, y_train, task_id)
        else:
            model = train_model(model, train_x, train_y)

        # ----------------------------------------------------
        # Update buffer
        # ----------------------------------------------------

        if method in ["reservoir","uncertainty","geometry"]:         
            replay_buffer.add_batch(x_train,y_train,task_id=task_id)        

        elif method == "clsampling":
            replay_buffer.add_batch(model, x_train, y_train,task_id)
        
        
        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------
        for eval_task in range(task_id + 1):
            _, _, x_eval, y_eval = all_task_data[eval_task]

            if method == "memory_efficient":
                preds = mer_model.predict(x_eval)
                mae = mean_absolute_error(y_eval, preds)
            else:
                mae = evaluate_model(model, x_eval, y_eval)

            results[task_id, eval_task] = mae     
            #print(f"Eval Task {eval_task+1} MAE: {mae:.4f}")


    # ========================================================
    # Final Metrics
    # ========================================================

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

    return {
        "results": results,
        "final_avg_mae": final_avg_mae,
        "final_avg_forgetting": final_avg_forgetting
    }

# ============================================================
# Multi-Seed Experiment Configuration
# ============================================================

# Every method is evaluated using exactly the same seeds.
# This makes the comparison reproducible and keeps the random
# conditions consistent across methods.

METHOD_CONFIGS = {

    # --------------------------------------------------------
    # No replay
    # --------------------------------------------------------
    #"none": {},

    # --------------------------------------------------------
    # Full replay
    # --------------------------------------------------------
    #"full": {},

    # --------------------------------------------------------
    # Reservoir replay
    # --------------------------------------------------------
    #"reservoir": {
    #    "memory_size": 55720,
    #    "replay_batch_size": 5500
    #},

    # --------------------------------------------------------
    # Uncertainty replay
    # --------------------------------------------------------
    #"uncertainty": {
    #    "memory_size": 336419,
    #    "replay_batch_size": 30000
    #},

    # --------------------------------------------------------
    # Geometry replay
    # --------------------------------------------------------
    "geometry": {
        "memory_size": 41681,
        "replay_batch_size": 4200
    },

    # --------------------------------------------------------
    # CLSampling replay
    # --------------------------------------------------------
    #"clsampling": {
    #    "memory_size": 30000,
    #    "replay_batch_size": 3000
    #},
 
    # --------------------------------------------------------
    # Memory Efficient Replay
    # -------------------------------------------------------- 
    # "memory_efficient": {}
}


# ============================================================
# Run One Seed
# ============================================================
 
def run_one_seed(method, seed, config):

    print("\n" + "=" * 80)
    print(f"METHOD = {method.upper()} | SEED = {seed}")
    print("=" * 80) 

    # IMPORTANT:
    # Set the seed before creating the model and replay buffer.
    set_seed(seed)

    output = continual_learning_experiment(
        method=method,
        **config
    )

    # Save the complete task-by-task result matrix.
    matrix_file = f"results_{method}_seed_{seed}.npy"
    np.save(matrix_file, output["results"])

    print(f"\nTask matrix saved to: {matrix_file}")

    print(
        f"Seed {seed} | "
        f"MAE = {output['final_avg_mae']:.6f} | "
        f"Forgetting = {output['final_avg_forgetting']:.6f}"
    )

    return {
        "method": method, 
        "seed": seed,
        "final_avg_mae": output["final_avg_mae"],
        "final_avg_forgetting": output["final_avg_forgetting"]
    }


# ============================================================
# Run All Seeds for All Methods
# ============================================================

all_results = []

for method, config in METHOD_CONFIGS.items():

    print("\n\n")
    print("#" * 80)
    print(f"STARTING METHOD: {method.upper()}")
    print("#" * 80)

    for seed in SEEDS:

        result = run_one_seed(
            method=method,
            seed=seed,
            config=config
        )

        all_results.append(result)


# ============================================================
# Individual Seed Results
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 80)
print("INDIVIDUAL SEED RESULTS")
print("=" * 80)

print(results_df.to_string(index=False))

results_df.to_csv(
    "multi_seed_results.csv",
    index=False
)


# ============================================================
# Mean and Standard Deviation
# ============================================================

summary_df = (
    results_df
    .groupby("method")
    .agg(
        MAE_mean=("final_avg_mae", "mean"),
        MAE_std=("final_avg_mae", "std"),

        Forgetting_mean=("final_avg_forgetting", "mean"),
        Forgetting_std=("final_avg_forgetting", "std")
    )
    .reset_index()
)


print("\n")
print("=" * 80)
print("FINAL RESULTS: MEAN ± STANDARD DEVIATION")
print("=" * 80)

for _, row in summary_df.iterrows():

    print(
        f"{row['method']:20s} | "
        f"MAE = {row['MAE_mean']:.4f} ± {row['MAE_std']:.4f} | "
        f"Forgetting = "
        f"{row['Forgetting_mean']:.4f} ± "
        f"{row['Forgetting_std']:.4f}"
    )


summary_df.to_csv(
    "multi_seed_summary.csv",
    index=False
)


# ============================================================
# Publication-Friendly Summary Table
# ============================================================

publication_df = summary_df.copy()

publication_df["MAE"] = publication_df.apply(
    lambda row:
        f"{row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}",
    axis=1
)

publication_df["Forgetting"] = publication_df.apply(
    lambda row:
        f"{row['Forgetting_mean']:.4f} ± "
        f"{row['Forgetting_std']:.4f}",
    axis=1
)

publication_df = publication_df[
    ["method", "MAE", "Forgetting"]
]

publication_df.to_csv(
    "publication_results.csv",
    index=False
)

print("\n")
print("=" * 80)
print("PUBLICATION TABLE")
print("=" * 80)

print(publication_df.to_string(index=False))

print("\nSaved:")
print("  multi_seed_results.csv")
print("  multi_seed_summary.csv")
print("  publication_results.csv")
print("  results_<method>_seed_<seed>.npy")



# ============================================================
# Plot Mean ± Standard Deviation Across Seeds
# ============================================================

def plot_multi_seed_summary(summary_df):

    # MAE
    plt.figure(figsize=(10, 6))

    x = np.arange(len(summary_df))

    plt.errorbar(
        x,
        summary_df["MAE_mean"],
        yerr=summary_df["MAE_std"],
        fmt="o",
        capsize=5
    )

    plt.xticks(
        x,
        summary_df["method"],
        rotation=30,
        ha="right"
    )

    plt.ylabel("Final Average MAE")
    plt.title("Final Average MAE Across Seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Forgetting
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        x,
        summary_df["Forgetting_mean"],
        yerr=summary_df["Forgetting_std"],
        fmt="o",
        capsize=5
    )

    plt.xticks(
        x,
        summary_df["method"],
        rotation=30,
        ha="right"
    )

    plt.ylabel("Final Average Forgetting")
    plt.title("Final Average Forgetting Across Seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Plotting
# ============================================================

def plot_results(results, title):

    plt.figure(figsize=(8, 6))
    plt.imshow(results,cmap="viridis",interpolation="nearest")
    plt.colorbar(label="MAE")
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