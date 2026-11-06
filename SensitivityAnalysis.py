import random
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from geometrybuffer import GeometryBuffer
from data.real.ettm2continual import ETTM2Dataset

SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

MEMORY_SIZE = 55720
REPLAY_BATCH_SIZE = 5500
EPOCHS = 50
LEARNING_RATE = 0.001

OUTPUT_DIR = Path("geometry_joint_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = ETTM2Dataset(
    file_path="data/real/ETTm2.csv",
    target_col="OT",
    seq_len=24,
)

TASKS = dataset.create_tasks()
total_train_samples = sum(len(task[1]) for task in TASKS)
total_test_samples = sum(len(task[3]) for task in TASKS)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Number of tasks       : {len(TASKS)}")
print(f"Total training samples: {total_train_samples}")
print(f"Total test samples    : {total_test_samples}")
print("=" * 60)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, x_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_tensor = torch.as_tensor(
        x_train, dtype=torch.float32, device=device
    )
    y_tensor = torch.as_tensor(
        y_train, dtype=torch.float32, device=device
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(
            x_test, dtype=torch.float32, device=device
        )
        predictions = model(x_tensor).cpu().numpy()
    return mean_absolute_error(y_test, predictions)


def compute_final_avg_mae(results):    
    final_row = results[-1]
    valid = np.isfinite(final_row)
    return float(np.mean(final_row[valid])) if valid.any() else np.nan


def compute_final_avg_forgetting(results):    
    num_tasks = results.shape[0]
    forgetting_scores = []

    for task_id in range(num_tasks - 1):
        task_history = results[task_id:, task_id]
        task_history = task_history[np.isfinite(task_history)]

        if len(task_history) < 2:
            continue

        best_mae = np.min(task_history[:-1])
        final_mae = results[-1, task_id]

        if np.isfinite(final_mae):
            forgetting_scores.append(final_mae - best_mae)

    return (
        float(np.mean(forgetting_scores))
        if forgetting_scores else np.nan
    )

def continual_learning_experiment(alpha, lambda_geom):
    input_dim = TASKS[0][1].shape[1]
    model = MLPRegressor(input_dim=input_dim).to(device)

    replay_buffer = GeometryBuffer(
        max_size=MEMORY_SIZE,
        alpha=alpha,
        lambda_geom=lambda_geom,
    )

    num_tasks = len(TASKS)   
    results = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    all_task_data = []

    for task_id, (task_name, x_train, y_train, x_test, y_test) in enumerate(TASKS):
        print(f"  Training task {task_id + 1}/{num_tasks}: {task_name}")

        all_task_data.append((x_train, y_train, x_test, y_test))

        train_x = x_train
        train_y = y_train       
        if task_id > 0:
            replay_x, replay_y = replay_buffer.sample(REPLAY_BATCH_SIZE)

            if replay_x is not None and replay_y is not None:
                train_x = np.concatenate([x_train, replay_x], axis=0)
                train_y = np.concatenate([y_train, replay_y], axis=0)

        model = train_model(model, train_x, train_y)        
        replay_buffer.add_batch(x_train, y_train, task_id=task_id)
        
        for eval_task in range(task_id + 1):
            _, _, x_eval, y_eval = all_task_data[eval_task]
            results[task_id, eval_task] = evaluate_model(
                model, x_eval, y_eval
            )

    return {
        "results": results,
        "final_avg_mae": compute_final_avg_mae(results),
        "final_avg_forgetting": compute_final_avg_forgetting(results),
    }


def run_one_configuration(alpha, lambda_geom, seed):
    print("\n" + "=" * 78)
    print(
        f"GEOMETRY JOINT SEARCH | alpha={alpha:.2f} | "
        f"lambda_geom={lambda_geom:.2f} | seed={seed}"
    )
    print("=" * 78)
    
    set_seed(seed)

    output = continual_learning_experiment(
        alpha=alpha,
        lambda_geom=lambda_geom,
    )

    matrix_path = OUTPUT_DIR / (
        f"task_matrix_alpha_{alpha:.2f}_lambda_{lambda_geom:.2f}_seed_{seed}.npy"
    )
    np.save(matrix_path, output["results"])

    result = {
        "alpha": alpha,
        "lambda_geom": lambda_geom,
        "seed": seed,
        "final_avg_mae": output["final_avg_mae"],
        "final_avg_forgetting": output["final_avg_forgetting"],
    }

    print(
        f"RESULT | MAE={result['final_avg_mae']:.6f} | "
        f"Forgetting={result['final_avg_forgetting']:.6f}"
    )
    return result


def main():
    all_results = []
    configurations = list(itertools.product(ALPHAS, LAMBDAS))

    print(f"Device: {device}")
    print(f"Parameter combinations: {len(configurations)}")
    print(f"Seeds per combination: {len(SEEDS)}")
    print(f"Total runs: {len(configurations) * len(SEEDS)}")
    print("Note: lambda_geom=0 makes alpha irrelevant to sampling,")
    print("but those rows are retained for a complete grid table.")

    for config_index, (alpha, lambda_geom) in enumerate(configurations, start=1):
        print(
            f"\nCONFIGURATION {config_index}/{len(configurations)}: "
            f"alpha={alpha:.2f}, lambda_geom={lambda_geom:.2f}"
        )

        for seed in SEEDS:
            result = run_one_configuration(alpha, lambda_geom, seed)
            all_results.append(result)

            # Save incrementally so results survive an interrupted run.
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "joint_individual_results.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        OUTPUT_DIR / "joint_individual_results.csv", index=False
    )

    summary_df = (
        results_df
        .groupby(["alpha", "lambda_geom"], as_index=False)
        .agg(
            MAE_mean=("final_avg_mae", "mean"),
            MAE_std=("final_avg_mae", "std"),
            Forgetting_mean=("final_avg_forgetting", "mean"),
            Forgetting_std=("final_avg_forgetting", "std"),
        )
    )
    summary_df.to_csv(OUTPUT_DIR / "joint_summary.csv", index=False)

    # Publication-friendly table.
    publication_df = summary_df.copy()
    publication_df["MAE"] = publication_df.apply(
        lambda row: f"{row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}",
        axis=1,
    )
    publication_df["Forgetting"] = publication_df.apply(
        lambda row: (
            f"{row['Forgetting_mean']:.4f} ± "
            f"{row['Forgetting_std']:.4f}"
        ),
        axis=1,
    )
    publication_df[
        ["alpha", "lambda_geom", "MAE", "Forgetting"]
    ].to_csv(OUTPUT_DIR / "joint_publication_table.csv", index=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS BY MEAN MAE")
    print("=" * 80)
    print(
        summary_df.sort_values("MAE_mean")
        .head(10)
        .to_string(index=False)
    )

    
    mae_grid = summary_df.pivot(
        index="alpha", columns="lambda_geom", values="MAE_mean"
    ).sort_index()
    plt.figure(figsize=(11, 7))
    sns.heatmap(mae_grid, annot=True, fmt=".4f", cmap="viridis_r")
    plt.title("Joint Hyperparameter Search: Mean MAE")
    plt.xlabel("lambda_geom")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mae_heatmap.png", dpi=200)
    plt.show()

    
    forgetting_grid = summary_df.pivot(
        index="alpha", columns="lambda_geom", values="Forgetting_mean"
    ).sort_index()
    plt.figure(figsize=(11, 7))
    sns.heatmap(forgetting_grid, annot=True, fmt=".4f", cmap="viridis_r")
    plt.title("Joint Hyperparameter Search: Mean Forgetting")
    plt.xlabel("lambda_geom")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forgetting_heatmap.png", dpi=200)
    plt.show()

    print("\nSaved outputs in:", OUTPUT_DIR.resolve())
    print("  joint_individual_results.csv")
    print("  joint_summary.csv")
    print("  joint_publication_table.csv")
    print("  mae_heatmap.png")
    print("  forgetting_heatmap.png")
    print("  task_matrix_*.npy")


if __name__ == "__main__":
    main()
