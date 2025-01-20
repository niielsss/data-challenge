import torch
import torch.nn as nn

class DeconvolutionModel(nn.Module):
    """
    A neural network model that performs image restoration using convolutional layers for encoding
    (downsampling) and deconvolution (transpose convolution) layers for decoding (upsampling).
    
    The model consists of two main components:
    1. Encoder: Downsamples the input image using Conv2d layers with ReLU activations.
    2. Decoder: Upsamples the feature maps using ConvTranspose2d layers, with the final layer producing
       an output with 3 channels (RGB).
    
    The model is designed to map low-resolution images to high-resolution outputs.

    Methods:
        forward(x): Defines the forward pass, taking the input tensor 'x' and passing it through the encoder
                    and decoder to produce the output.
    """
    def __init__(self):
        """
        Initializes the DeconvolutionModel. Defines the encoder and decoder parts of the network.

        The encoder reduces the spatial dimensions of the input image, and the decoder restores
        the spatial dimensions back to the original image size.
        """
        super(DeconvolutionModel, self).__init__()
        
        # Encoder (downsampling) using Conv2d layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder (upsampling) using ConvTranspose2d layers (deconvolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        The input tensor 'x' is first passed through the encoder layers to extract features,
        followed by the decoder layers to generate the output image.

        Args:
            x (torch.Tensor): Input tensor (image) with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input (batch_size, 3, height, width).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x