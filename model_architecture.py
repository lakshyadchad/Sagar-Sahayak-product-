
#this is a copy file for dashboard and original one is in ../model/model_architecture.py

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()        # --- HELPER 1: The Down-Sampling Block (The Encoder) ---
        # This shrinks the image size but increases the "depth" (feature understanding).
        # Example: Input 256x256 -> Output 128x128
        def down_block(in_feat, out_feat, normalize=True):
            layers: list[nn.Module] = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)        # --- HELPER 2: The Up-Sampling Block (The Decoder) ---
        # This grows the image back up.
        # Example: Input 128x128 -> Output 256x256
        def up_block(in_feat, out_feat, dropout=0.0):
            layers: list[nn.Module] = [
                # Transpose Convolution is the mathematical opposite of Convolution (it makes things bigger)
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout)) # Prevents overfitting
            return nn.Sequential(*layers)
            
            # --- ENCODER LAYERS ---
        # Input: 256 x 256 x 3
        
        # Layer 1: Output 128 x 128 x 64
        self.down1 = down_block(3, 64, normalize=False) 
        
        # Layer 2: Output 64 x 64 x 128
        self.down2 = down_block(64, 128)
        
        # Layer 3: Output 32 x 32 x 256
        self.down3 = down_block(128, 256)
        
        # Layer 4: Output 16 x 16 x 512
        self.down4 = down_block(256, 512)
        
        # Layer 5: Output 8 x 8 x 512 (The Bottom of the U - Bottleneck)
        self.down5 = down_block(512, 512)

        # --- DECODER LAYERS ---
        
        # Up 1: Takes input from down5. Output 16 x 16 x 512.
        self.up1 = up_block(512, 512, dropout=0.5)
        
        # Up 2: Takes input from up1 AND down4 (Skip Connection). 
        # Input channels = 512 (from up1) + 512 (from down4) = 1024
        self.up2 = up_block(1024, 256)
        
        # Up 3: Takes input from up2 AND down3. 
        # Input = 256 + 256 = 512
        self.up3 = up_block(512, 128)
        
        # Up 4: Takes input from up3 AND down2.
        # Input = 128 + 128 = 256
        self.up4 = up_block(256, 64)
        
        # --- FINAL OUTPUT LAYER ---
        # Takes input from up4 AND down1.
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), # 128 -> 256
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # Collapse back to 3 channels (RGB)
            nn.Tanh() # Squashes output to range [-1, 1] (Standard for Images)
        )

    def forward(self, x):
        # 1. GOING DOWN (Save the outputs d1, d2... for later!)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4) # The bottleneck

        # 2. GOING UP (Concatenate with saved outputs)
        
        # First upsample
        u1 = self.up1(d5)
        
        # Skip Connection 1: Join u1 with d4
        # torch.cat([u1, d4], 1) means "stack them along the channel axis"
        u1_concat = torch.cat([u1, d4], 1) 
        u2 = self.up2(u1_concat)
        
        # Skip Connection 2: Join u2 with d3
        u2_concat = torch.cat([u2, d3], 1)
        u3 = self.up3(u2_concat)
        
        # Skip Connection 3: Join u3 with d2
        u3_concat = torch.cat([u3, d2], 1)
        u4 = self.up4(u3_concat)
        
        # 3. FINAL IMAGE RECONSTRUCTION
        # Last Skip Connection: Join u4 with d1 (The original high-res details)
        u4_concat = torch.cat([u4, d1], 1)
        
        return self.final(u4_concat)
