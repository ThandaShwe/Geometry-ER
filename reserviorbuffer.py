import numpy as np
import random

class ReservoirBuffer:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.buffer = []
        self.n_seen = 0

    def __len__(self):
        return len(self.buffer)    

    def add(self,x,y,task_id=None):
        self.n_seen += 1
        sample = {
            "x": x,
            "y": y,
            "task_id": task_id
        }
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            j = random.randint(
                0,
                self.n_seen - 1
            )
            if j < self.max_size:
                self.buffer[j] = sample

    
    def add_batch(
            self,
            X,
            Y,
            task_id=None):
        print("In the add batch method")
        for x, y in zip(X, Y):

            self.add(
                x,
                y,
                task_id
            )
        print("Added Buffer Size:", len(X))    

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None

        batch_size = min(
            batch_size,
            len(self.buffer)
        )
        samples = random.sample(
            self.buffer,
            batch_size
        )
        X = np.vstack([
            s["x"] for s in samples
        ])
        Y = np.vstack([
            s["y"] for s in samples
        ])
        return X, Y