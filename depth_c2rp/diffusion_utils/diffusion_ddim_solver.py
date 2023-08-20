import torch
import numpy as np
                    
def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start , beta_end , num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def ddpm_steps(x, seq, model, b, sde, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            mean = 1 / torch.sqrt(1 - beta_t) * (x - beta_t * output / torch.sqrt(1 - at))
            
            x0_preds.append(mean)
            
            noise = torch.randn_like(x).to(x.device)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample)
    return xs, xs

def generalized_steps(x, seq, model, b, sde, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x] 
        for idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = compute_alpha(b, t.long())            
            at_next = compute_alpha(b, next_t.long())
#            at = sde.alpha(t)
#            at_next = sde.alpha(next_t)
#            print("at", at)
            
            xt = xs[-1].to(x.device)
            
            #print("xt.shape", xt.shape)
            #print("t", t.shape)
            
            et = model(xt, t) 
            #print("et", et.shape)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            #x0_preds.append(x0_t.to('cpu'))
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 1.0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x).to(x.device) + c2 * et
            #print("xt_next.shape", xt_next.shape)
            #xs.append(xt_next.to('cpu'))
            xs.append(xt_next)
    
    #print("xs", xs[-1][0])
    #print("x0_preds", x0_preds[-1][0]) 

    return xs, xs

