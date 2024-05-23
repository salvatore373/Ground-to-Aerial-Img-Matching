import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf, n_layers=4):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)

# Example
input_nc = 3  # Numero di canali dell'immagine di input (es. immagine terrestre)
output_nc = 3  # Numero di canali dell'immagine di output (es. immagine satellitare)
ndf = 64  # Numero di filtri nella prima convoluzione
n_layers = 4  # Numero di strati

discriminator = Discriminator(input_nc, output_nc, ndf, n_layers)
print(discriminator)
