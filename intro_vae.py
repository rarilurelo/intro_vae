from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))


def l_reg(mu, std):
    return - 0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2, dim=-1)

def loss_function(x, x_r,
                z_mu, z_std,
                z_r_mu, z_r_std,
                z_pp_mu, z_pp_std,
                z_r_detach_mu, z_r_detach_std,
                z_pp_detach_mu, z_pp_detach_std):
        l_ae = torch.sum((x.reshape(-1, 784) - x_r.reshape(-1, 784)) ** 2, dim=-1)
        l_e_adv = l_reg(z_mu, z_std) + alpha * (F.relu(m - l_reg(z_r_detach_mu, z_r_detach_std)) + F.relu(m - l_reg(z_pp_detach_mu, z_pp_detach_std)))
        l_g_adv = alpha * (l_reg(z_r_mu, z_r_std) + l_reg(z_pp_mu, z_pp_std))
        loss = torch.mean(l_e_adv + l_g_adv + beta * l_ae)
        return loss

alpha = 0.5
beta = 0.5
m = 0.5

encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0002)


def train(epoch):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        x = data
        x = x.reshape(-1, 784)
        optimizer.zero_grad()

        z_mu, z_logvar = encoder(x)
        z_std = torch.exp(0.5*z_logvar)
        eps = torch.randn_like(z_std)
        z = eps.mul(z_std).add_(z_mu)

        x_r = decoder(z)
        z_r_mu, z_r_logvar = encoder(x_r)
        z_r_std = torch.exp(0.5*z_r_logvar)
        eps = torch.randn_like(z_r_std)
        z_r = eps.mul(z_r_std).add_(z_r_mu)

        z_r_detach_mu, z_r_detach_logvar = encoder(x_r.detach())
        z_r_detach_std = torch.exp(0.5*z_r_detach_logvar)
        eps = torch.randn_like(z_r_detach_std)
        z_r_detach = eps.mul(z_r_detach_std).add_(z_r_detach_mu)

        z_p = torch.randn_like(z)
        x_p = decoder(z_p)

        z_pp_mu, z_pp_logvar = encoder(x_p)
        z_pp_std = torch.exp(0.5*z_pp_logvar)
        eps = torch.randn_like(z_pp_std)
        z_pp = eps.mul(z_pp_std).add_(z_pp_mu)

        z_pp_detach_mu, z_pp_detach_logvar = encoder(x_p.detach())
        z_pp_detach_std = torch.exp(0.5*z_pp_detach_logvar)
        eps = torch.randn_like(z_pp_detach_std)
        z_pp_detach = eps.mul(z_pp_detach_std).add_(z_pp_detach_mu)

        loss = loss_function(x, x_r,
                z_mu, z_std,
                z_r_mu, z_r_std,
                z_pp_mu, z_pp_std,
                z_r_detach_mu, z_r_detach_std,
                z_pp_detach_mu, z_pp_detach_std)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            x = data
            x = x.reshape(-1, 784)

            z_mu, z_logvar = encoder(x)
            z_std = torch.exp(0.5*z_logvar)
            eps = torch.randn_like(z_std)
            z = eps.mul(z_std).add_(z_mu)

            x_r = decoder(z)
            z_r_mu, z_r_logvar = encoder(x_r)
            z_r_std = torch.exp(0.5*z_r_logvar)
            eps = torch.randn_like(z_r_std)
            z_r = eps.mul(z_r_std).add_(z_r_mu)

            z_r_detach_mu, z_r_detach_logvar = encoder(x_r.detach())
            z_r_detach_std = torch.exp(0.5*z_r_detach_logvar)
            eps = torch.randn_like(z_r_detach_std)
            z_r_detach = eps.mul(z_r_detach_std).add_(z_r_detach_mu)

            z_p = torch.randn_like(z)
            x_p = decoder(z_p)

            z_pp_mu, z_pp_logvar = encoder(x_p)
            z_pp_std = torch.exp(0.5*z_pp_logvar)
            eps = torch.randn_like(z_pp_std)
            z_pp = eps.mul(z_pp_std).add_(z_pp_mu)

            z_pp_detach_mu, z_pp_detach_logvar = encoder(x_p.detach())
            z_pp_detach_std = torch.exp(0.5*z_pp_detach_logvar)
            eps = torch.randn_like(z_pp_detach_std)
            z_pp_detach = eps.mul(z_pp_detach_std).add_(z_pp_detach_mu)

            loss = loss_function(x, x_r,
                    z_mu, z_std,
                    z_r_mu, z_r_std,
                    z_pp_mu, z_pp_std,
                    z_r_detach_mu, z_r_detach_std,
                    z_pp_detach_mu, z_pp_detach_std)

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = decoder(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')

