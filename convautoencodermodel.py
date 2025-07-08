import torch.nn as nn

class ConvAutoencoder64(nn.Module):
    def __init__(self):
        super(ConvAutoencoder64, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

class ConvAutoencoder128(nn.Module):
    def __init__(self):
        super(ConvAutoencoder128, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

class ConvAutoencoder256(nn.Module):
    def __init__(self):
        super(ConvAutoencoder256, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # 256 -> 128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), # 4 -> 2
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), # 2 -> 1
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1), # 1 -> 2
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1), # 2 -> 4
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1), # 4 -> 8
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), # 8 -> 16
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 16 -> 32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 64 -> 128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),    # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)