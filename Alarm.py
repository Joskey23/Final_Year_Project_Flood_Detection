#!/usr/bin/env python
"""
GUI-Free Flood Detection Alarm System
Uses Attention U-Net model to detect floods with zero GUI dependencies
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - NO GUI
import matplotlib.pyplot as plt
from PIL import Image
import pygame
import warnings
from datetime import datetime
from pathlib import Path
import csv
import rasterio
from skimage import exposure, morphology
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize pygame for audio (no display needed)
pygame.mixer.init()

#########################################
# Attention U-Net Model Definition
#########################################

class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class ConvBlock(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """Upsampling block with attention for U-Net"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpBlock, self).__init__()
        self.attention = AttentionGate(F_g=in_ch//2, F_l=in_ch//2, F_int=in_ch//4)
        
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
            
        self.conv = ConvBlock(in_ch, out_ch)
        
    def forward(self, x1, x2):
        # x1 is from encoder, x2 is from previous decoder stage
        x2 = self.up(x2)
        
        # Handle potential size mismatch
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention mechanism
        x1 = self.attention(g=x2, x=x1)
        
        # Concatenate
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class AttentionUNet(nn.Module):
    """Attention U-Net for water detection"""
    def __init__(self, n_channels, n_classes, filters_base=64):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder path
        self.inc = ConvBlock(n_channels, filters_base)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(filters_base, filters_base*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(filters_base*2, filters_base*4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(filters_base*4, filters_base*8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(filters_base*8, filters_base*16)
        )
        
        # Decoder path with attention
        self.up4 = UpBlock(filters_base*16, filters_base*8)
        self.up3 = UpBlock(filters_base*8, filters_base*4)
        self.up2 = UpBlock(filters_base*4, filters_base*2)
        self.up1 = UpBlock(filters_base*2, filters_base)
        
        # Output layer
        self.outc = nn.Conv2d(filters_base, n_classes, kernel_size=1)
        
        # Auxiliary outputs for deep supervision
        self.aux_out3 = nn.Conv2d(filters_base*4, n_classes, kernel_size=1)
        self.aux_out2 = nn.Conv2d(filters_base*2, n_classes, kernel_size=1)
        self.aux_out1 = nn.Conv2d(filters_base, n_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Record input size for upsampling later
        input_size = (x.size()[2], x.size()[3])
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with attention gates
        x = self.up4(x4, x5)
        x = self.up3(x3, x)
        aux3 = self.aux_out3(x)
        
        x = self.up2(x2, x)
        aux2 = self.aux_out2(x)
        
        x = self.up1(x1, x)
        aux1 = self.aux_out1(x)
        
        # Main output
        main_output = self.outc(x)
        
        # Resize all outputs to match input size
        aux3 = F.interpolate(aux3, size=input_size, mode='bilinear', align_corners=True)
        aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=True)
        
        # Return main output and auxiliary outputs during training
        if self.training:
            # Structure it to be compatible with our loss function
            return main_output, (aux1, aux2, aux3)
        else:
            # Return only the main output for inference
            return main_output

#########################################
# Utility Functions
#########################################

def create_beep_sound(output_path="alarm.wav"):
    """Create a simple beep sound file for the alarm"""
    try:
        import scipy.io.wavfile as wav
        
        # Create a simple beep sound
        sample_rate = 44100
        duration = 0.5  # seconds
        frequency = 440.0  # Hz (A4 note)
        
        # Generate beep
        t = np.linspace(0, duration, int(sample_rate * duration))
        beep = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Make it louder
        beep = np.clip(beep * 3.0, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        beep = (beep * 32767).astype(np.int16)
        
        # Create a pulsing effect by repeating the beep
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.int16)
        full_sound = np.concatenate([beep, silence, beep, silence, beep])
        
        # Save to file
        wav.write(output_path, sample_rate, full_sound)
        
        print(f"Created alarm sound file at {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating alarm sound: {e}")
        return False


def get_arr_flood(fname):
    """Load GeoTIFF data with error handling using only rasterio (no OpenCV)"""
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return None
    
    try:
        with rasterio.open(fname) as src:
            # Read data
            arr = src.read()  # Read all bands (shape: [C, H, W])
            
            # Apply appropriate scale factors if available
            if hasattr(src, 'scales'):
                for i, scale in enumerate(src.scales):
                    if scale != 1:
                        arr[i] = arr[i] * scale
                        
            # Check for and replace NaN values
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr)
                        
        return arr
    except Exception as e:
        print(f"Error reading file {fname}: {e}")
        return None


def calculate_ndwi(nir_band, green_band, method='ndwi'):
    """
    Calculate water indices for improved water detection
    """
    # Avoid division by zero
    valid_mask = np.logical_and(np.isfinite(nir_band), np.isfinite(green_band))
    valid_mask = np.logical_and(valid_mask, (nir_band + green_band) != 0)
    
    water_index = np.zeros_like(nir_band, dtype=np.float32)
    
    if method.lower() == 'ndwi':
        # Standard NDWI (McFeeters, 1996): (Green-NIR)/(Green+NIR)
        water_index[valid_mask] = (green_band[valid_mask] - nir_band[valid_mask]) / (green_band[valid_mask] + nir_band[valid_mask])
    
    elif method.lower() == 'mndwi':
        # Modified NDWI: (Green-SWIR)/(Green+SWIR)
        water_index[valid_mask] = (green_band[valid_mask] - nir_band[valid_mask]) / (green_band[valid_mask] + nir_band[valid_mask])
    
    elif method.lower() == 'ndpi' and nir_band.shape == green_band.shape:
        # For SAR: (VV-VH)/(VV+VH)
        water_index[valid_mask] = (nir_band[valid_mask] - green_band[valid_mask]) / (nir_band[valid_mask] + green_band[valid_mask])
    
    else:
        # Default to standard NDWI
        water_index[valid_mask] = (green_band[valid_mask] - nir_band[valid_mask]) / (green_band[valid_mask] + nir_band[valid_mask])
    
    # Clip values to [-1, 1] range
    water_index = np.clip(water_index, -1, 1)
    
    # Scale to [0, 1] for model input
    water_index = (water_index + 1) / 2
    
    return water_index


def sar_normalize(image, clip_min=-50, clip_max=1, enhance_small_features=True):
    """Normalize SAR image with appropriate clipping and scaling"""
    # Clip values to sensible range for SAR
    clipped = np.clip(image, clip_min, clip_max)
    
    # Normalize to [0, 1]
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    
    # Apply enhancement for small water features if requested
    if enhance_small_features:
        return enhance_sar_for_small_features(normalized)
    else:
        return normalized


def enhance_sar_for_small_features(image):
    """Enhanced preprocessing specifically for small water features in SAR images"""
    # Check if input has channels dimension
    if len(image.shape) == 3:
        # Assume [C, H, W] format
        if image.shape[0] <= 4:  # Reasonable number of channels
            enhanced = np.zeros_like(image)
            for i in range(image.shape[0]):
                enhanced[i] = enhance_sar_channel(image[i])
            return enhanced
        else:  # Assume [H, W, C] format
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = enhance_sar_channel(image[:, :, i])
            return enhanced
    else:
        # Single channel
        return enhance_sar_channel(image)


def enhance_sar_channel(channel):
    """Enhance a single SAR channel for better water feature detection"""
    # 1. Adaptive contrast enhancement
    p2, p98 = np.percentile(channel, (2, 98))
    image_rescale = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # 2. Bilateral filtering to preserve edges while reducing noise
    try:
        from skimage.restoration import denoise_bilateral
        denoised = denoise_bilateral(image_rescale, sigma_spatial=1.5, 
                                     sigma_color=0.1, win_size=5)
    except ImportError:
        from scipy.ndimage import median_filter
        denoised = median_filter(image_rescale, size=3)
    
    # 3. Edge enhancement for better water boundaries
    try:
        from skimage import filters
        edges = filters.sobel(denoised)
        # Add weighted edges to enhance boundaries
        enhanced = denoised + 0.08 * edges
        enhanced = np.clip(enhanced, 0, 1)
    except ImportError:
        enhanced = denoised
    
    # 4. Local adaptive histogram equalization for further contrast boost
    try:
        enhanced = exposure.equalize_adapthist(enhanced, clip_limit=0.02)
    except:
        pass
    
    return enhanced


def apply_morphological_cleaning(mask, min_size=50):
    """Apply morphological operations to clean up the prediction"""
    # Remove small objects
    cleaned = morphology.remove_small_objects(mask, min_size=min_size)
    # Close holes
    cleaned = morphology.closing(cleaned, morphology.disk(2))
    # Remove small holes
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    
    return cleaned


def multi_scale_inference(model, img, device, num_classes, scales=[0.75, 1.0, 1.25, 1.5], use_flip=True):
    """Run inference at multiple scales and merge results for better accuracy"""
    # Convert to torch tensor if numpy array
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    
    # Add batch dimension if needed
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    
    img = img.to(device)
    
    # Base image size
    b, _, h, w = img.shape
    
    # Initialize output probabilities
    final_prob = torch.zeros((b, num_classes, h, w), device=device)
    
    # Run inference at different scales
    for scale in scales:
        # Scale the image
        if scale != 1.0:
            scaled_img = F.interpolate(img, scale_factor=scale, 
                                      mode='bilinear', align_corners=True)
        else:
            scaled_img = img
        
        # Regular forward pass
        with torch.no_grad():
            output = model(scaled_img)
            # Output is already the main output for AttentionUNet in eval mode
            
            # Scale back to original size
            if scale != 1.0:
                output = F.interpolate(output, size=(h, w), 
                                     mode='bilinear', align_corners=True)
            
            prob = F.softmax(output, dim=1)
            final_prob += prob
        
        # Horizontal flip augmentation
        if use_flip:
            # Flip the image
            flipped_img = torch.flip(scaled_img, dims=[3])
            
            # Forward pass on flipped image
            with torch.no_grad():
                flipped_output = model(flipped_img)
                
                # Scale back to original size
                if scale != 1.0:
                    flipped_output = F.interpolate(flipped_output, size=(h, w), 
                                                 mode='bilinear', align_corners=True)
                
                # Flip back the output
                flipped_output = torch.flip(flipped_output, dims=[3])
                
                prob = F.softmax(flipped_output, dim=1)
                final_prob += prob
    
    # Average the probabilities
    div_factor = len(scales) * (2 if use_flip else 1)
    final_prob /= div_factor
    
    return final_prob


def load_data_from_csv(csv_path, s1_dir, label_dir, optical_dir=None, jrc_dir=None, is_flood=True, max_samples=None):
    """Load data from CSV file with S1Hand, LabelHand, and optionally JRCWaterHand"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    
    # Read CSV file
    data_pairs = []
    try:
        # Read the CSV using pandas if available, otherwise use csv module
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            columns = df.columns.tolist()
            
            # Process each row
            for _, row in df.iterrows():
                s1_file = row[columns[0]]  # S1Hand file
                label_file = row[columns[1]]  # LabelHand file
                jrc_file = row[columns[2]] if len(columns) > 2 else None  # JRCWaterHand file if available
                
                # Construct absolute paths
                s1_path = os.path.join(s1_dir, s1_file)
                label_path = os.path.join(label_dir, label_file)
                
                # Get base name from S1 file (without extension and suffix)
                base_name = s1_file.split('_S1Hand')[0]
                
                # Construct optical path using the same basename
                optical_path = None
                if optical_dir:
                    optical_file = f"{base_name}_S2Hand.tif"
                    optical_path = os.path.join(optical_dir, optical_file)
                
                # Construct JRC water path
                jrc_path = None
                if jrc_dir and jrc_file:
                    jrc_path = os.path.join(jrc_dir, jrc_file)
                
                # Check if required files exist
                if os.path.exists(s1_path) and os.path.exists(label_path):
                    data_pairs.append((s1_path, label_path, optical_path, jrc_path, is_flood))
        except:
            # Fallback to csv module
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                columns = next(reader, None)  # Get header or first row
                
                if not columns:
                    print(f"CSV file is empty: {csv_path}")
                    return []
                
                # Process each row
                for row in reader:
                    if len(row) < 2:
                        continue  # Skip invalid rows
                    
                    s1_file = row[0]  # S1Hand file
                    label_file = row[1]  # LabelHand file
                    jrc_file = row[2] if len(row) > 2 else None  # JRCWaterHand file if available
                    
                    # Construct absolute paths
                    s1_path = os.path.join(s1_dir, s1_file)
                    label_path = os.path.join(label_dir, label_file)
                    
                    # Get base name from S1 file (without extension and suffix)
                    base_name = s1_file.split('_S1Hand')[0]
                    
                    # Construct optical path using the same basename
                    optical_path = None
                    if optical_dir:
                        optical_file = f"{base_name}_S2Hand.tif"
                        optical_path = os.path.join(optical_dir, optical_file)
                    
                    # Construct JRC water path
                    jrc_path = None
                    if jrc_dir and jrc_file:
                        jrc_path = os.path.join(jrc_dir, jrc_file)
                    
                    # Check if required files exist
                    if os.path.exists(s1_path) and os.path.exists(label_path):
                        data_pairs.append((s1_path, label_path, optical_path, jrc_path, is_flood))
                
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
    
    # Limit samples if requested
    if max_samples is not None and len(data_pairs) > max_samples:
        import random
        random.shuffle(data_pairs)
        data_pairs = data_pairs[:max_samples]
    
    print(f"Loaded {len(data_pairs)} samples from {csv_path}")
    return data_pairs


def analyze_water_content(label_path, jrc_path=None, is_flood=True, boost_perm_water=True):
    """
    Analyze water content in a label mask, distinguishing flood from permanent water
    with improved handling of JRC permanent water
    """
    try:
        # Load label mask
        label_arr = get_arr_flood(label_path)
        if label_arr is None:
            return None
            
        # Load JRC permanent water if available
        jrc_arr = None
        if jrc_path and os.path.exists(jrc_path) and boost_perm_water:
            jrc_arr = get_arr_flood(jrc_path)
            
            # Ensure JRC mask is 2D
            if jrc_arr is not None and len(jrc_arr.shape) > 2:
                jrc_arr = jrc_arr.squeeze()
                
            # Convert to binary mask
            if jrc_arr is not None:
                jrc_arr = (jrc_arr > 0).astype(np.uint8)
            
        # Ensure label is 2D
        if len(label_arr.shape) > 2:
            label_arr = label_arr.squeeze()
        
        # Convert -1 values to 255 (no data) in labels
        label_arr[label_arr == -1] = 255
        
        # Get a modifiable copy
        label_arr_modified = label_arr.copy()
        
        # If JRC water is available, use it to identify permanent water
        if jrc_arr is not None and boost_perm_water:
            # Where JRC shows water, mark as permanent water (class 1)
            # Only do this for valid pixels (not in no-data regions)
            valid_mask = (label_arr != 255)
            # For pixels that are marked as water in labels and also in JRC,
            # force these to be permanent water
            water_mask = (label_arr == 1) & valid_mask
            perm_water_mask = jrc_arr & water_mask
            
            # Update the label array - this will make some flood pixels into perm water
            label_arr_modified[perm_water_mask] = 1
        
        # Check if we're using multi-class scenario
        num_classes = 3  # Default
        
        # For multi-class scenario (with class 1 = perm water, class 2 = flood)
        if num_classes > 2:
            if is_flood:
                # If it's a flood sample, water pixels not marked as perm are flood (class 2)
                water_mask = (label_arr_modified == 1)
                valid_mask = (label_arr_modified != 255)
                
                # Manually separate permanent water using JRC
                if jrc_arr is not None and boost_perm_water:
                    # Pixels that are water in both label and JRC are perm water
                    perm_water_mask = jrc_arr & water_mask
                    # Remaining water pixels are flood
                    flood_mask = water_mask & (~perm_water_mask)
                    
                    # Mark in the label array
                    label_arr_modified[perm_water_mask] = 1  # Permanent water
                    label_arr_modified[flood_mask] = 2       # Flood water
                else:
                    # Without JRC, assume all water pixels in flood samples are flood
                    label_arr_modified[water_mask] = 2  # All to flood class
                
                # Count pixel types
                perm_pixels = np.sum(perm_water_mask) if jrc_arr is not None else 0
                flood_pixels = np.sum(water_mask) - perm_pixels
                water_pixels = perm_pixels + flood_pixels
                valid_pixels = np.sum(valid_mask)
                
            else:
                # If it's a permanent water sample, water pixels are permanent water (class 1)
                water_mask = (label_arr_modified == 1)
                valid_mask = (label_arr_modified != 255)
                
                water_pixels = np.sum(water_mask)
                valid_pixels = np.sum(valid_mask)
                flood_pixels = 0  # No flood in permanent water samples
                perm_pixels = water_pixels
        else:
            # Binary classification (simpler case)
            water_mask = (label_arr_modified == 1)
            valid_mask = (label_arr_modified != 255)
            
            water_pixels = np.sum(water_mask)
            valid_pixels = np.sum(valid_mask)
            
            # For binary, still track separate statistics even though not used in model
            if is_flood:
                if jrc_arr is not None and boost_perm_water:
                    perm_pixels = np.sum(jrc_arr & water_mask)
                    flood_pixels = water_pixels - perm_pixels
                else:
                    flood_pixels = water_pixels
                    perm_pixels = 0
            else:
                flood_pixels = 0
                perm_pixels = water_pixels
        
        # Skip samples with no valid pixels
        if valid_pixels == 0:
            return None
        
        # Calculate percentages
        water_percent = water_pixels / valid_pixels if valid_pixels > 0 else 0
        flood_percent = flood_pixels / valid_pixels if valid_pixels > 0 else 0
        perm_percent = perm_pixels / valid_pixels if valid_pixels > 0 else 0
        
        return {
            'water_pixels': water_pixels,
            'flood_pixels': flood_pixels,
            'perm_pixels': perm_pixels,
            'valid_pixels': valid_pixels,
            'water_percent': water_percent,
            'flood_percent': flood_percent,
            'perm_percent': perm_percent,
            'modified_label': label_arr_modified  # Return the modified label array
        }
        
    except Exception as e:
        print(f"Error analyzing {label_path}: {e}")
        return None



def filter_by_water_content(data_pairs, min_water_percent=0.01):
    """Filter data pairs by minimum water percentage"""
    filtered_pairs = []
    
    for data_pair in tqdm(data_pairs, desc="Filtering by water content"):
        img_path, label_path, optical_path, jrc_path, is_flood = data_pair
        stats = analyze_water_content(label_path, jrc_path, is_flood, boost_perm_water=True)
        if stats and stats['water_percent'] >= min_water_percent:
            filtered_pairs.append((img_path, label_path, optical_path, jrc_path, is_flood, stats))
    
    print(f"Filtered to {len(filtered_pairs)} samples with at least {min_water_percent:.1%} water content")
    return filtered_pairs


def preprocess_multi_source(s1_path, optical_path=None, jrc_path=None, 
                          patch_size=None, include_ndwi=True, include_jrc=True,
                          water_index='ndwi'):
    """
    Preprocess multiple data sources (S1, optical, JRC) for model input - without OpenCV
    """
    try:
        # Load S1 SAR image with rasterio
        s1_arr = get_arr_flood(s1_path)
        if s1_arr is None:
            print(f"Failed to load SAR image: {s1_path}")
            return None, None
        
        # Normalize SAR data
        s1_arr = sar_normalize(s1_arr, clip_min=-50, clip_max=1, enhance_small_features=True)
        
        # Initialize combined_input with SAR data
        if len(s1_arr.shape) == 3 and s1_arr.shape[0] == 2:
            # Already in [C, H, W] format
            combined_input = s1_arr.copy()
        else:
            # Convert to [C, H, W] format if needed
            if len(s1_arr.shape) == 3 and s1_arr.shape[2] == 2:
                # [H, W, C] to [C, H, W]
                combined_input = s1_arr.transpose(2, 0, 1).copy()
            elif len(s1_arr.shape) == 2:
                # Single channel - create two identical channels
                combined_input = np.stack([s1_arr, s1_arr]).copy()
            else:
                print(f"Unexpected SAR data shape: {s1_arr.shape}")
                return None, None
        
        # Get dimensions
        _, height, width = combined_input.shape
        
        # Create a copy of SAR for visualization
        if combined_input.shape[0] == 2:
            # Create an RGB visualization from VV and VH channels
            sar_vis = np.zeros((height, width, 3), dtype=np.float32)
            sar_vis[:, :, 0] = combined_input[0]  # VV -> Red
            sar_vis[:, :, 1] = combined_input[1]  # VH -> Green
            sar_vis[:, :, 2] = (combined_input[0] + combined_input[1]) / 2  # Average -> Blue
        else:
            # Grayscale image
            sar_vis = np.zeros((height, width, 3), dtype=np.float32)
            sar_vis[:, :, 0] = combined_input[0]
            sar_vis[:, :, 1] = combined_input[0]
            sar_vis[:, :, 2] = combined_input[0]
        
        # Load and process S2Hand optical data for NDWI if requested and available
        if include_ndwi and optical_path and os.path.exists(optical_path):
            optical_arr = get_arr_flood(optical_path)
            if optical_arr is not None:
                # For Sentinel-2 data, assuming standard band order:
                # Band 3 is Green, Band 8 is NIR
                green_band = optical_arr[2] if optical_arr.shape[0] >= 3 else None
                nir_band = optical_arr[7] if optical_arr.shape[0] >= 8 else None
                
                if nir_band is not None and green_band is not None:
                    # Calculate NDWI
                    ndwi = calculate_ndwi(nir_band, green_band, method=water_index)
                    
                    # Resize NDWI if needed (using PIL instead of OpenCV)
                    if ndwi.shape != (height, width):
                        # Convert to PIL image
                        ndwi_pil = Image.fromarray((ndwi * 255).astype(np.uint8))
                        ndwi_pil = ndwi_pil.resize((width, height), Image.BILINEAR)
                        # Convert back to numpy
                        ndwi = np.array(ndwi_pil).astype(np.float32) / 255.0
                    
                    # Add to combined input
                    combined_input = np.vstack([combined_input, ndwi[np.newaxis, :, :]])
                else:
                    # No suitable bands - create a dummy NDWI channel
                    combined_input = np.vstack([combined_input, np.zeros((1, height, width), dtype=np.float32)])
        elif include_ndwi:
            # Create a synthetic NDWI from SAR
            vv = combined_input[0]
            vh = combined_input[1]
            pseudo_ndwi = (vv - vh) / (vv + vh + 1e-8)
            pseudo_ndwi = (pseudo_ndwi + 1) / 2  # Scale to [0, 1]
            combined_input = np.vstack([combined_input, pseudo_ndwi[np.newaxis, :, :]])
        
        # Load and process JRC permanent water data if requested and available
        if include_jrc and jrc_path and os.path.exists(jrc_path):
            jrc_arr = get_arr_flood(jrc_path)
            if jrc_arr is not None:
                # Ensure JRC mask is 2D
                if len(jrc_arr.shape) > 2:
                    jrc_arr = jrc_arr.squeeze()
                
                # Normalize JRC data to [0, 1] and amplify signal
                jrc_arr = (jrc_arr > 0).astype(np.float32) * 5.0
                
                # Resize JRC mask if needed (using PIL instead of OpenCV)
                if jrc_arr.shape != (height, width):
                    # Convert to PIL image
                    jrc_pil = Image.fromarray((jrc_arr * 51).astype(np.uint8))  # Scale to 0-255
                    jrc_pil = jrc_pil.resize((width, height), Image.NEAREST)
                    # Convert back to numpy and rescale 
                    jrc_arr = np.array(jrc_pil).astype(np.float32) / 51.0 * 5.0
                
                # Add to combined input
                combined_input = np.vstack([combined_input, jrc_arr[np.newaxis, :, :]])
        elif include_jrc:
            # Create a synthetic JRC water mask (constant zeros)
            combined_input = np.vstack([combined_input, np.zeros((1, height, width), dtype=np.float32)])
        
        # Resize if requested (using PIL instead of OpenCV)
        if patch_size is not None:
            # Resize all channels
            resized_input = np.zeros((combined_input.shape[0], patch_size, patch_size), dtype=np.float32)
            for i in range(combined_input.shape[0]):
                # Convert to PIL image
                channel_pil = Image.fromarray((combined_input[i] * 255).astype(np.uint8))
                channel_pil = channel_pil.resize((patch_size, patch_size), Image.BILINEAR)
                # Convert back to numpy
                resized_input[i] = np.array(channel_pil).astype(np.float32) / 255.0
            
            # Resize visualization image
            sar_vis_pil = Image.fromarray((sar_vis * 255).astype(np.uint8))
            sar_vis_pil = sar_vis_pil.resize((patch_size, patch_size), Image.BILINEAR)
            sar_vis = np.array(sar_vis_pil).astype(np.float32) / 255.0
            
            combined_input = resized_input
        
        # Convert to torch tensor
        tensor = torch.from_numpy(combined_input).float()
        
        # Add batch dimension if it doesn't exist
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        return tensor, sar_vis
    
    except Exception as e:
        print(f"Error preprocessing multi-source data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def apply_colormap(prediction, num_classes=3, original_image=None, alpha=0.6):
    """
    Apply colormap to prediction mask without using OpenCV
    """
    # Define colors for each class: [background, perm water, flood]
    colors = [
        [0, 0, 0],        # Background: black
        [0, 0, 255],      # Permanent water: blue
        [255, 0, 0]       # Flood: red
    ]
    
    # Create RGB visualization
    rgb_pred = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    # Assign colors to each class
    for cls_idx, color in enumerate(colors):
        if cls_idx < num_classes:
            mask = prediction == cls_idx
            rgb_pred[mask, 0] = color[0]
            rgb_pred[mask, 1] = color[1]
            rgb_pred[mask, 2] = color[2]
    
    # Overlay on original image if provided
    if original_image is not None:
        # Ensure original image has right shape and dtype
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
        
        # Resize original image if needed using PIL
        if original_image.shape[:2] != rgb_pred.shape[:2]:
            # Convert to PIL image
            orig_pil = Image.fromarray(original_image)
            orig_pil = orig_pil.resize((rgb_pred.shape[1], rgb_pred.shape[0]), Image.BILINEAR)
            # Convert back to numpy
            original_image = np.array(orig_pil)
        
        # Create mask for non-background pixels
        mask = (prediction > 0)
        
        # Create blended image
        blended = original_image.copy()
        for c in range(3):  # For each color channel
            blended[:, :, c] = np.where(
                mask, 
                (1-alpha) * original_image[:, :, c] + alpha * rgb_pred[:, :, c],
                original_image[:, :, c]
            )
        
        return blended
    else:
        return rgb_pred


def calculate_flood_stats(prediction, num_classes=3):
    """
    Calculate statistics about flood pixels
    """
    # Count pixels for each class
    total_pixels = prediction.size
    background_pixels = np.sum(prediction == 0)
    
    if num_classes > 2:
        # Multi-class case
        perm_water_pixels = np.sum(prediction == 1)
        flood_pixels = np.sum(prediction == 2)
        
        # Calculate percentages
        perm_water_percent = (perm_water_pixels / total_pixels) * 100
        flood_percent = (flood_pixels / total_pixels) * 100
        total_water_percent = ((perm_water_pixels + flood_pixels) / total_pixels) * 100
        
        return {
            'total_pixels': total_pixels,
            'background_pixels': background_pixels,
            'perm_water_pixels': perm_water_pixels,
            'flood_pixels': flood_pixels,
            'perm_water_percent': perm_water_percent,
            'flood_percent': flood_percent,
            'total_water_percent': total_water_percent
        }
    else:
        # Binary case
        water_pixels = np.sum(prediction == 1)
        water_percent = (water_pixels / total_pixels) * 100
        
        return {
            'total_pixels': total_pixels,
            'background_pixels': background_pixels,
            'water_pixels': water_pixels,
            'water_percent': water_percent
        }


def save_visualization(image, save_path):
    """Save visualization to file using matplotlib (no GUI needed)"""
    try:
        # Make sure we're using the Agg backend
        plt.switch_backend('Agg')
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Visualization saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return False


def add_alert_text(image):
    """Add alert text to image without using OpenCV"""
    # Create a copy of the image
    alert_img = image.copy()
    
    # Create a red alert banner at the top
    banner_height = 40
    banner = np.zeros((banner_height, alert_img.shape[1], 3), dtype=np.uint8)
    banner[:, :, 0] = 255  # Red color (R channel)
    
    # Add text using PIL instead of OpenCV
    try:
        # Convert banner to PIL image
        from PIL import ImageDraw, ImageFont
        banner_pil = Image.fromarray(banner)
        draw = ImageDraw.Draw(banner_pil)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 10), "FLOOD ALERT!", fill=(255, 255, 255), font=font)
        
        # Convert back to numpy
        banner = np.array(banner_pil)
    except Exception as e:
        # If PIL text drawing fails, just use a colored banner
        print(f"Warning: Could not add text to alert banner: {e}")
    
    # Concatenate banner with image
    result = np.vstack([banner, alert_img])
    
    return result


#########################################
# Main Flood Detector Class
#########################################

class FloodDetector:
    """Core class for flood detection"""
    
    def __init__(self, args):
        """
        Initialize the flood detector with args
        
        Args:
            args: Arguments from command line
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model
        print(f"Loading model from {args.checkpoint_path}...")
        print(f"Using device: {self.device}")
        
        # Determine input channels
        self.input_channels = 2  # Base: VV and VH bands
        if args.include_ndwi:
            self.input_channels += 1
        if args.include_jrc:
            self.input_channels += 1
        
        # Initialize model
        self.model = AttentionUNet(
            n_channels=self.input_channels,
            n_classes=args.num_classes,
            filters_base=args.filters_base
        )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Standard checkpoint format
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Simple state dict format
                self.model.load_state_dict(checkpoint)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Setup alarm sound
        if not os.path.exists("alarm.wav"):
            create_beep_sound()
    
    def process_sample(self, s1_path, label_path=None, optical_path=None, jrc_path=None, stats=None):
        """
        Process a single sample with all available data sources
        
        Args:
            s1_path: Path to SAR image
            label_path: Optional path to label mask
            optical_path: Optional path to optical image
            jrc_path: Optional path to JRC water mask
            stats: Optional pre-computed water statistics
            
        Returns:
            Dictionary with detection results
        """
        try:
            print(f"Processing: {os.path.basename(s1_path)}")
            
            # Preprocess image
            input_tensor, original_image = preprocess_multi_source(
                s1_path,
                optical_path=optical_path,
                jrc_path=jrc_path,
                patch_size=self.args.patch_size,
                include_ndwi=self.args.include_ndwi,
                include_jrc=self.args.include_jrc,
                water_index=self.args.water_index
            )
            
            if input_tensor is None:
                print("Failed to preprocess image")
                return None
            
            # Check input shape
            if input_tensor.shape[1] != self.input_channels:
                print(f"Warning: Input has {input_tensor.shape[1]} channels, but model expects {self.input_channels}")
                if input_tensor.shape[1] < self.input_channels:
                    # Pad with zeros
                    padding = torch.zeros(
                        (input_tensor.shape[0], self.input_channels - input_tensor.shape[1], 
                        input_tensor.shape[2], input_tensor.shape[3]), 
                        device=input_tensor.device
                    )
                    input_tensor = torch.cat([input_tensor, padding], dim=1)
                else:
                    # Truncate
                    input_tensor = input_tensor[:, :self.input_channels]
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                if self.args.use_multiscale:
                    # Multi-scale inference
                    outputs = multi_scale_inference(
                        self.model, 
                        input_tensor, 
                        self.device, 
                        self.args.num_classes, 
                        scales=[1.0, 1.25] if self.args.use_multiscale else [1.0], 
                        use_flip=True
                    )
                    probs = outputs  # Already probabilities
                else:
                    # Standard inference
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
            
            # Get class predictions
            preds = torch.argmax(probs, dim=1)
            
            # Apply morphological post-processing if enabled
            if self.args.use_morphological_postprocessing:
                # Process each sample in the batch
                processed_preds = []
                for i in range(preds.size(0)):
                    # Convert to numpy for morphological operations
                    pred_np = preds[i].cpu().numpy()
                    
                    # Process each class separately
                    cleaned_pred = np.zeros_like(pred_np)
                    for c in range(1, self.args.num_classes):  # Skip background class
                        class_mask = (pred_np == c)
                        cleaned_mask = apply_morphological_cleaning(class_mask, min_size=50)
                        cleaned_pred[cleaned_mask] = c
                    
                    # Convert back to tensor
                    processed_preds.append(torch.from_numpy(cleaned_pred).to(self.device))
                
                # Stack back to batch
                preds = torch.stack(processed_preds, dim=0)
            
            # Convert to numpy for further processing
            preds_np = preds.squeeze().cpu().numpy()
            probs_np = probs.squeeze().cpu().numpy()
            
            # Calculate statistics
            stats = calculate_flood_stats(preds_np, self.args.num_classes)
            
            # Create visualization
            colored_result = apply_colormap(
                preds_np, 
                self.args.num_classes, 
                original_image=original_image
            )
            
            # Prepare results
            results = {
                'prediction': preds_np,
                'probabilities': probs_np,
                'visualization': colored_result,
                'stats': stats,
                'original_image': original_image
            }
            
            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(os.path.basename(s1_path))[0]
            
            # Save visualization
            save_path = os.path.join(self.args.output_dir, f"{basename}_{timestamp}.png")
            save_visualization(colored_result, save_path)
            
            # Print results
            self._print_stats(stats)
            
            # Check if flood detected (for alarm)
            self._check_flood_threshold(stats, basename, colored_result, timestamp)
            
            return results
            
        except Exception as e:
            print(f"Error processing sample {s1_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_dataset(self, dataset='test'):
        """
        Process a specific dataset (train, val, test)
        
        Args:
            dataset: Which dataset to process ('train', 'val', 'test')
        """
        print(f"Processing {dataset} dataset...")
        
        # Determine CSV path based on dataset
        data_root = self.args.data_root
        csv_path = os.path.join(
            data_root, 
            f"splits/flood_handlabeled/flood_{dataset}_data_with_jrc.csv"
        )
        
        # Load data pairs
        data_pairs = load_data_from_csv(
            csv_path,
            self.args.s1_dir,
            self.args.label_dir,
            optical_dir=self.args.optical_dir if self.args.include_ndwi else None,
            jrc_dir=self.args.jrc_dir if self.args.include_jrc else None,
            is_flood=True
        )
        
        # Filter by water content if needed
        if self.args.min_water_percent > 0:
            data_pairs = filter_by_water_content(data_pairs, self.args.min_water_percent)
        
        # Limit to num_samples if specified
        if self.args.num_samples is not None and self.args.num_samples < len(data_pairs):
            import random
            random.shuffle(data_pairs)
            data_pairs = data_pairs[:self.args.num_samples]
        
        # Process each sample
        for i, (s1_path, label_path, optical_path, jrc_path, is_flood) in enumerate(
            tqdm(data_pairs, desc=f"Processing {dataset} samples")):
            
            self.process_sample(s1_path, label_path, optical_path, jrc_path)
            
            # Wait for user input before continuing to the next sample
            if i < len(data_pairs) - 1:  # Don't wait after the last image
                self._wait_for_user_continue()
    
    def _check_flood_threshold(self, stats, basename, colored_result, timestamp):
        """Check if flood percentage exceeds threshold and trigger alarm if needed"""
        # Set an alarm threshold (percentage of flood pixels)
        threshold = 5.0  # Default threshold of 5%
        
        if self.args.num_classes > 2:
            # For multi-class, check flood water percentage
            flood_percent = stats['flood_percent']
            exceeded = flood_percent >= threshold
        else:
            # For binary, check total water percentage
            water_percent = stats['water_percent']
            exceeded = water_percent >= threshold
        
        if exceeded:
            print(f"FLOOD ALERT! Threshold exceeded: {threshold:.1f}%")
            
            # Add alert text to image and save an alarm version
            alert_img = add_alert_text(colored_result)
            alert_path = os.path.join(self.args.output_dir, f"{basename}_ALERT_{timestamp}.png")
            save_visualization(alert_img, alert_path)
            
            # Trigger alarm sound
            self._trigger_alarm()
            
            return True
        else:
            print(f"No flood alert (threshold: {threshold:.1f}%)")
            return False
    
    def _trigger_alarm(self):
        """Play alarm sound"""
        try:
            # Play alarm sound
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.play()
            
            # Print alert
            print("\n" + "!" * 50)
            print("!!!           FLOOD ALERT           !!!")
            print("!" * 50 + "\n")
            
            # Let alarm sound play for 3 seconds
            time.sleep(3)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error playing alarm: {e}")

    def _wait_for_user_continue(self):
        """Wait for user input to continue processing the next image"""
        print("\n" + "-" * 50)
        print("Press Enter to continue to the next image or 'q' to quit...")
        user_input = input()
        if user_input.lower() == 'q':
            print("Stopping evaluation as requested by user.")
            sys.exit(0)
        print("-" * 50 + "\n")

    def _print_stats(self, stats):
        """Print detection statistics"""
        print("\nDetection Results:")
        if self.args.num_classes > 2:
            print(f"Permanent Water: {stats['perm_water_percent']:.2f}%")
            print(f"Flood Water: {stats['flood_percent']:.2f}%")
            print(f"Total Water: {stats['total_water_percent']:.2f}%")
        else:
            print(f"Water: {stats['water_percent']:.2f}%")
        print("")


#########################################
# Main Function
#########################################

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GUI-Free Attention U-Net Flood Detection')
    
    # Model parameters
    parser.add_argument('--interactive', action='store_true',
                  help='Wait for user confirmation after each image')
    parser.add_argument('--checkpoint_path', type=str, 
                      default="/mnt/z/results_attn_unet_flooddetector/checkpoints/AttentionUNet_HandLabeled_NDWI_JRC_20250414_124655_best.pth", 
                      help='Path to the model checkpoint')
    parser.add_argument('--data_root', type=str, 
                      default="/mnt/z/Formal data_v1.1", 
                      help='Root data directory')
    parser.add_argument('--optical_dir', type=str, 
                      default="/mnt/z/Formal data_v1.1/data/flood_events/HandLabeled/S2Hand", 
                      help='Directory containing optical imagery')
    parser.add_argument('--s1_dir', type=str, 
                      default="/mnt/z/Formal data_v1.1/data/flood_events/HandLabeled/S1Hand", 
                      help='Directory containing SAR imagery')
    parser.add_argument('--label_dir', type=str, 
                      default="/mnt/z/Formal data_v1.1/data/flood_events/HandLabeled/LabelHand", 
                      help='Directory containing label masks')
    parser.add_argument('--jrc_dir', type=str, 
                      default="/mnt/z/Formal data_v1.1/data/flood_events/HandLabeled/JRCWaterHand", 
                      help='Directory containing JRC permanent water')
    parser.add_argument('--dataset', type=str, 
                      default='test', choices=['train', 'val', 'test'], 
                      help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, 
                      default=8, 
                      help='Batch size for inference')
    parser.add_argument('--patch_size', type=int, 
                      default=256, 
                      help='Patch size for evaluation')
    parser.add_argument('--num_classes', type=int, 
                      default=3, 
                      help='Number of classes (0:background, 1:perm_water, 2:flood)')
    parser.add_argument('--include_ndwi', type=bool, 
                      default=True, 
                      help='Include NDWI as additional channel')
    parser.add_argument('--include_jrc', type=bool, 
                      default=True, 
                      help='Include JRC permanent water as additional channel')
    parser.add_argument('--use_crf', type=bool, 
                      default=True, 
                      help='Use CRF for post-processing')
    parser.add_argument('--water_index', type=str, 
                      default='ndwi', choices=['ndwi', 'mndwi', 'ndpi'], 
                      help='Water index calculation method')
    parser.add_argument('--min_water_percent', type=float, 
                      default=0.00, 
                      help='Minimum water percentage to include sample')
    parser.add_argument('--output_dir', type=str, 
                      default='/mnt/z/eval_results_UNET_ALARM', 
                      help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, 
                      default=50, 
                      help='Number of samples to visualize')
    parser.add_argument('--use_multiscale', type=bool, 
                      default=True, 
                      help='Use multi-scale inference')
    parser.add_argument('--use_morphological_postprocessing', type=bool, 
                      default=True, 
                      help='Apply morphological post-processing')
    parser.add_argument('--filters_base', type=int, 
                      default=64, 
                      help='Base filters for the Attention U-Net')
    
    # Additional options for flexibility
    parser.add_argument('--image_path', type=str, 
                      help='Path to a single image to process (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize flood detector
    detector = FloodDetector(args)
    
    # Determine the mode based on arguments
    if args.image_path:
        # Process a single image
        detector.process_sample(args.image_path)
    else:
        # Process dataset
        detector.process_dataset(args.dataset)


if __name__ == "__main__":
    main()