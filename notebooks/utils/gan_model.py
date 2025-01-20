import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        """
        Initializes the Generator model.

        The Generator is designed to transform underwater images (input) 
        into clean images (output). It uses an encoder-decoder architecture, 
        where the encoder captures spatial features by downsampling, and the 
        decoder reconstructs the cleaned image by upsampling.
        """
        super(Generator, self).__init__()

        # Encoder: Downsamples the input image to a lower resolution.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder: Upsamples the feature maps back to the original resolution.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Defines the forward pass of the Generator.
        
        Args:
            x (torch.Tensor): Input tensor representing an underwater image, 
                              with shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: Output tensor representing the restored clean image, 
                          with the same shape as the input.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
