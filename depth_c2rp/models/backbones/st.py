from torch_scatter import scatter
import torch
import time

if __name__ == "__main__":
    input_ = torch.randn(5098, 64).cuda()
    idx = torch.randint(0, 152, (5098,)).cuda()
    print(input_.shape)
    print(idx.shape)
    
    for i in range(1000):
        t1 = time.time()
        scatter(input_, idx, dim=0, reduce='max')
        print(time.time() - t1)