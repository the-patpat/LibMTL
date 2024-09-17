import numpy as np
import matlab.engine
import time
import io
import psutil
import torch 
from pympler import muppy, summary
# from memory_profiler import profile

def generate_vectors_nd(n=3, use_torch=False):
    # Generate two random vectors in R3
    if use_torch:
        vector1 = torch.randn(n)
        vector2 = torch.randn(n)
        while torch.arccos(vector1.dot(vector2)/(torch.norm(vector1)*torch.norm(vector2))) >= np.pi/2:
            vector2 = torch.randn(n)
        vector1.to('cuda')
        vector2.to('cuda')
    else:
        vector1 = np.random.rand(n)
        vector2 = np.random.rand(n)

        # Ensure the angle between the vectors is less than 90 degrees
        while np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))) >= np.pi / 2:
            vector2 = np.random.rand(n)

    return vector1, vector2


# @profile
def par(v1, v2, n_channels, part_size):
    for i in range(10):
        x, exitflag = eng.solve_socp_parallel(
                v1,v2,
                n_channels, part_size,
                0.0, nargout=2, stdout=io.StringIO(), stderr=io.StringIO()
            )
        new_grads = [torch.Tensor(v2.copy()).squeeze()]
        y = np.asarray(x).copy()
        print(y.shape)
        new_grads.insert(0, 
                             torch.tensor(y).reshape(-1,).to(device=vector1.device))
        do_something(torch.stack(new_grads))
    return x, exitflag

# @profile
def do_something(t):
    t.sum()

if __name__ == "__main__":
    print("Starting matlab engine...", end='')
    eng = matlab.engine.start_matlab()
    # p = eng.gcp()
    print("done.")
    for i in range(1000):
        n_channels = 10
        part_size = 2000
        n = n_channels*part_size
        # n = 1000
        t1 = time.time()
        vector1, vector2 = generate_vectors_nd(n, False)
        # x, exitflag = eng.solve_socp(
        #     vector1.to('cpu').numpy().astype(float).reshape(-1,1),
        #     vector2.to('cpu').numpy().astype(float).reshape(-1,1),
        #     n, 0.0, nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
        # v1 = vector1.to('cpu').numpy().copy().astype(float).reshape(-1,1)
        # v2 = vector2.to('cpu').numpy().copy().astype(float).reshape(-1,1)
        x, exitflag = eng.solve_socp_parallel(
                vector1.reshape(-1,1),vector2.reshape(-1,1),
                n_channels, part_size,
                0.0, nargout=2, stdout=io.StringIO(), stderr=io.StringIO()
        )
        new_grads = [vector2.reshape(1,-1)]
        x = np.asarray(x).copy().reshape(1,-1)
        new_grads.insert(0, 
                             x)#torch.tensor(x).reshape(-1,).to(device=vector1.device))
        do_something(np.vstack(new_grads))
        mem_usage = psutil.virtual_memory()
        print(f'{i}\t{n}\t{(time.time() - t1):.2f}\t{mem_usage.used*1e-9:.2f}')
        # summary.print_(summary.summarize(muppy.get_objects()))
        # if i % 100 == 0:
            # print(eng.workspace)
    eng.exit()