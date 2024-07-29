import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from losses import  loss_fn
from models import *
from utils import *

from time import time
from termcolor import colored

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device = device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device = device)
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs =   50#@param {'type':'integer'}
## size of a mini-batch
batch_size =  32 #@param {'type':'integer'}
## learning rate
lr=1e-4 #@param {'type':'number'}

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 0) # num_workers=4
optimizer = Adam(score_model.parameters(), lr=lr)
# tqdm_epoch = tqdm.tqdm(range(n_epochs)) # tqdm.notebook.trange(n_epochs)

for epoch in range(n_epochs): # tqdm_epoch:
  t = time()
  print(colored('-' * (100 + 50 + 3 + len(str(n_epochs))), 'cyan'))
  print(colored('{} Epoch {}{} / {} {}'.format('-' * 70, ' ' * (2 - len(str(epoch + 1))), epoch + 1, n_epochs, '-' * 70), 'cyan'))
  print(colored('-' * (100 + 50 + 3 + len(str(n_epochs))), 'cyan'))
  avg_loss = 0.
  num_items = 0
  # print('PRIVET!!!!')
  loop = tqdm(data_loader, leave=True)
  for x, y in loop: #  data_loader:
    x = x.to(device)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
    loop.set_postfix(loss=loss.item())

  t = int(time() - t)
  t_min, t_sec = str(t // 60), str(t % 60)
  print(colored('Average Loss: {}'.format(round(avg_loss / num_items, 5)), 'cyan'))
  print(
    colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec), 'cyan')
  )
#   print(colored('-' * (60 + 50 + 3 + len(str(n_epochs))), 'cyan'))
  print()
  print()

  # Print the averaged training loss so far.
  # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), 'ckpt.pth')