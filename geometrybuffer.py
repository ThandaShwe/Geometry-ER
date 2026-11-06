
import numpy as np

from reserviorbuffer import ReservoirBuffer

class GeometryBuffer(ReservoirBuffer):    

    def __init__(self,max_size=500,alpha=0.7 ,lambda_geom=0.9, epsilon=1e-6):
        super().__init__(max_size)
        self.alpha = alpha
        self.lambda_geom = lambda_geom
        self.epsilon = epsilon
    
    #Geometry scores    
    def compute_geometry_scores(self):
        # Sort samples by input coordinates
        sorted_buffer = sorted(self.buffer,key=lambda s: s["x"][0])
        n = len(sorted_buffer)      
        if n < 3:
            scores = np.ones(n)
            return sorted_buffer, scores        

        x = np.array([s["x"][0] for s in sorted_buffer])
        y = np.array([s["y"][0] for s in sorted_buffer])
        # Compute slopes

        dx = np.diff(x) + self.epsilon
        dy = np.diff(y)
        slopes = dy / dx

        # Compute curvature

        curvature = np.abs(slopes[1:] - slopes[:-1])   

        scores = np.zeros(n)
        scores[1:-1] = curvature        
        scores[0] = curvature[0]
        scores[-1] = curvature[-1]   
        scores += self.epsilon
        scores = np.log1p(scores)
        return sorted_buffer, scores

    # Calculate probabilites 

    def compute_probabilities(self, scores):
            geom_probs = scores ** self.alpha
            geom_probs /= (geom_probs.sum() + self.epsilon)
            # uniform replay component 
            uniform_probs = np.ones_like(geom_probs) / len(geom_probs)
            # hybrid replay, geometry and uniform
            probs = (self.lambda_geom * geom_probs +(1 - self.lambda_geom)* uniform_probs)
            # Normalize again
            probs /= probs.sum()
            return probs

       

    def sample(self, batch_size):       

            if len(self.buffer) == 0:
                return None, None
            #Geometry scores
            sorted_buffer, scores = (self.compute_geometry_scores())
            # replay probabilites

            probs = self.compute_probabilities(scores)
            indices = np.random.choice(len(sorted_buffer), size=min(batch_size,len(sorted_buffer)),
                replace=False,
                p=probs
            )          

            samples = [sorted_buffer[i] for i in indices]
            X = np.vstack([s["x"] for s in samples])
            Y = np.vstack([s["y"] for s in samples])
            return X, Y

