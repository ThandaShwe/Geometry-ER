import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def task1(x):
    return np.sin(x)


def task2(x):
    return np.sin(x) + 0.3 * x


def task3(x):
    return 0.2 * x**2


def task4(x):
    return np.sin(3 * x)


def task5(x):
    return np.where(x < 0, np.sin(x), 0.5 * x**2)


TASKS = [
    ("Task1_Sine", task1),
    ("Task2_ShiftedSine", task2),
    ("Task3_Quadratic", task3),
    ("Task4_HighFreq", task4),
    ("Task5_Piecewise", task5),
]


def generate_task_data(
    task_fn,
    n_samples=500,
    x_range=(-5, 5),
    noise_std=0.1,
    rng=None,
):
    """Generate noisy synthetic regression data with shape (N, 1)."""
    if rng is None:
        rng = np.random.default_rng()

    x = rng.uniform(x_range[0], x_range[1], size=n_samples)
    y = task_fn(x).astype(np.float64)
    y = y + rng.normal(0.0, noise_std, size=n_samples)

    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def create_synthetic_tasks(
    n_train=500,
    n_test=200,
    x_range=(-5, 5),
    noise_std=0.1,
    seed=SEED,
):
    """Return tasks as (name, x_train, y_train, x_test, y_test)."""
    rng = np.random.default_rng(seed)
    tasks = []

    for name, task_fn in TASKS:
        x_train, y_train = generate_task_data(
            task_fn, n_samples=n_train, x_range=x_range,
            noise_std=noise_std, rng=rng
        )
        x_test, y_test = generate_task_data(
            task_fn, n_samples=n_test, x_range=x_range,
            noise_std=noise_std, rng=rng
        )
        tasks.append((name, x_train, y_train, x_test, y_test))

    return tasks


def visualize_tasks():
    x = np.linspace(-5, 5, 1000)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, (name, fn) in enumerate(TASKS):
        axes[idx].plot(x, fn(x))
        axes[idx].set_title(name)
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("y")
        axes[idx].grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.tight_layout()
    plt.show()


class ReservoirBuffer:    
    def __init__(self, max_size, seed=SEED):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self.x_memory = []
        self.y_memory = []
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x_memory)

    def add_batch(self, x, y, task_id=None):
        del task_id  
        for xi, yi in zip(x, y):
            self.n_seen += 1

            if len(self.x_memory) < self.max_size:
                self.x_memory.append(np.array(xi, copy=True))
                self.y_memory.append(np.array(yi, copy=True))
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.max_size:
                    self.x_memory[j] = np.array(xi, copy=True)
                    self.y_memory[j] = np.array(yi, copy=True)

    def sample(self, batch_size):
        if len(self) == 0 or batch_size <= 0:
            return None, None

        n = min(int(batch_size), len(self))
        indices = self.rng.choice(len(self), size=n, replace=False)
        x = np.stack([self.x_memory[i] for i in indices]).astype(np.float32)
        y = np.stack([self.y_memory[i] for i in indices]).astype(np.float32)
        return x, y


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

def train_model(model, x_train, y_train, epochs=50, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)

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
        x_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        predictions = model(x_tensor).cpu().numpy()
    return mean_absolute_error(y_test, predictions)


def compute_final_avg_mae(results):   
    final_row = results[-1]
    valid = np.isfinite(final_row)
    return float(np.mean(final_row[valid])) if np.any(valid) else float("nan")


def compute_final_avg_forgetting(results):   
    num_tasks = results.shape[0]
    forgetting_scores = []
    for task_id in range(num_tasks - 1):        
        historical = results[task_id:num_tasks - 1, task_id]
        historical = historical[np.isfinite(historical)]
        final_mae = results[-1, task_id]
        if historical.size and np.isfinite(final_mae):
            forgetting_scores.append(final_mae - np.min(historical))
    return float(np.mean(forgetting_scores)) if forgetting_scores else 0.0


def continual_learning_experiment(
    method="none",
    tasks=None,
    memory_size=500,
    replay_batch_size=128,
    epochs=50,
    lr=0.001,
    buffer_seed=SEED,
):
    
    method = method.lower()
    if method not in {"none", "full", "reservoir"}:
        raise ValueError("method must be 'none', 'full', or 'reservoir'")

    if tasks is None:
        tasks = create_synthetic_tasks()

    print("=" * 60)
    print(f"METHOD: {method.upper()}")
    print("=" * 60)

    input_dim = tasks[0][1].shape[1]
    model = MLPRegressor(input_dim=input_dim).to(device)
    replay_buffer = ReservoirBuffer(memory_size, seed=buffer_seed) if method == "reservoir" else None

    num_tasks = len(tasks)    
    results = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
    seen_task_data = []

    for task_id, (task_name, x_train, y_train, x_test, y_test) in enumerate(tasks):
        print(f"\nTraining on {task_name}")
        seen_task_data.append((x_train, y_train, x_test, y_test))

        train_x = x_train
        train_y = y_train
        replay_count = 0

        if method == "full":
            train_x = np.concatenate([item[0] for item in seen_task_data], axis=0)
            train_y = np.concatenate([item[1] for item in seen_task_data], axis=0)

        elif method == "reservoir":
            if task_id > 0:
                replay_x, replay_y = replay_buffer.sample(replay_batch_size)
                if replay_x is not None:
                    replay_count = len(replay_x)
                    train_x = np.concatenate([x_train, replay_x], axis=0)
                    train_y = np.concatenate([y_train, replay_y], axis=0)

        print(f"  Current-task samples: {len(x_train)}")
        print(f"  Training samples:     {len(train_x)}")
        if method == "reservoir":
            print(f"  Replay samples:       {replay_count}")

        model = train_model(model, train_x, train_y, epochs=epochs, lr=lr)        
        if method == "reservoir":
            replay_buffer.add_batch(x_train, y_train, task_id=task_id)
            print(f"  Buffer size:          {len(replay_buffer)}/{memory_size}")

        
        for eval_task_id in range(task_id + 1):
            _, _, x_eval, y_eval = seen_task_data[eval_task_id]
            mae = evaluate_model(model, x_eval, y_eval)
            results[task_id, eval_task_id] = mae
            print(f"  MAE on Task {eval_task_id + 1}: {mae:.4f}")

    final_avg_mae = compute_final_avg_mae(results)
    final_avg_forgetting = compute_final_avg_forgetting(results)

    print("\nFinal results")
    print(f"  Final average MAE:       {final_avg_mae:.4f}")
    print(f"  Final average forgetting: {final_avg_forgetting:.4f}")

    with open("final_results.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"{method.upper()} FINAL RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Final Average MAE: {final_avg_mae:.4f}\n")
        f.write(f"Final Average Forgetting: {final_avg_forgetting:.4f}\n")

    print("Results appended to final_results.txt")
    return results



def plot_results(results, title, task_names=None):
    if task_names is None:
        task_names = [f"T{i + 1}" for i in range(results.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(results)
    image = ax.imshow(masked, cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="MAE")
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Training task")
    ax.set_title(title)
    ax.set_xticks(range(len(task_names)), task_names, rotation=30, ha="right")
    ax.set_yticks(range(len(task_names)), task_names)
    fig.tight_layout()
    plt.show()


def plot_method_comparison(results_by_method):    
    task_names = [name for name, *_ in TASKS]
    x = np.arange(len(task_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (method, results) in enumerate(results_by_method.items()):
        ax.bar(x + (idx - (len(results_by_method) - 1) / 2) * width,
               results[-1], width, label=method)

    ax.set_xticks(x, task_names, rotation=25, ha="right")
    ax.set_ylabel("Final MAE (lower is better)")
    ax.set_title("Final performance by task and replay method")
    ax.legend()
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    set_seed(SEED)    
    synthetic_tasks = create_synthetic_tasks(
        n_train=800,
        n_test=200,
        x_range=(-5, 5),
        noise_std=0.1,
        seed=SEED,
    )

    print("Synthetic dataset")
    print(f"Number of tasks: {len(synthetic_tasks)}")
    print(f"Training samples per task: {len(synthetic_tasks[0][1])}")
    print(f"Test samples per task: {len(synthetic_tasks[0][3])}")
    print(f"Device: {device}")
    
    # visualize_tasks()

    results_by_method = {}

    for method in ("none", "full", "reservoir"):
        set_seed(SEED)
        results_by_method[method] = continual_learning_experiment(
            method=method,
            tasks=synthetic_tasks,
            memory_size=800,
            replay_batch_size=256,
            epochs=50,
            lr=0.001,
            buffer_seed=SEED,
        )

    task_names = [name for name, *_ in synthetic_tasks]
    plot_results(results_by_method["none"], "No Replay", task_names)
    plot_results(results_by_method["full"], "Full Replay", task_names)
    plot_results(results_by_method["reservoir"], "Reservoir Replay", task_names)
    plot_method_comparison(results_by_method)
    visualize_tasks()
