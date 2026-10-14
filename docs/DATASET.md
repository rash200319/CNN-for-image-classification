# EuroSAT RGB Dataset Specification & Verification Profile

This document details the dataset preparation and statistical verification for **EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification**.

## 1. Overview
- **Dataset Name**: EuroSAT (RGB Variant, Sentinel-2 Satellite Imagery)
- **Source Platform**: European Space Agency (ESA) Copernicus Sentinel-2 / Zenodo
- **Total Images**: 27,000
- **Number of Classes**: 10
- **Native Spatial Resolution**: 64 x 64 pixels (Satisfies the $\le 64 \times 64$ constraint)
- **Color Channels**: 3 (RGB)
- **Reproducibility Seed**: 42

## 2. Per-Class Sample Distribution
| Class Name | Sample Count | Percentage (%) |
|---|---:|---:|
| AnnualCrop | 3,000 | 11.11% |
| Forest | 3,000 | 11.11% |
| HerbaceousVegetation | 3,000 | 11.11% |
| Highway | 2,500 | 9.26% |
| Industrial | 2,500 | 9.26% |
| Pasture | 2,000 | 7.41% |
| PermanentCrop | 2,500 | 9.26% |
| Residential | 3,000 | 11.11% |
| River | 2,500 | 9.26% |
| SeaLake | 3,000 | 11.11% |
| **Total** | **27,000** | **100.00%** |

## 3. Stratified Partition Distribution (70% / 15% / 15%)
| Split Partition | Target Ratio | Sample Count | Actual Percentage (%) |
|---|---:|---:|---:|
| Training | 70% | 18,900 | 70.00% |
| Validation | 15% | 4,049 | 15.00% |
| Testing | 15% | 4,051 | 15.00% |

## 4. Channel-Wise Pixel Normalization Statistics
Empirically computed over RGB normalized space $[0.0, 1.0]$:
- **Mean (R, G, B)**: `[0.3412, 0.3787, 0.4054]`
- **Standard Deviation (R, G, B)**: `[0.0924, 0.0659, 0.0561]`

*Note: Custom models (Model A and Model B) use standard $(0.5, 0.5, 0.5)$ centering mapping inputs to $[-1.0, 1.0]$, while fine-tuned SOTA models use canonical ImageNet statistics.*
