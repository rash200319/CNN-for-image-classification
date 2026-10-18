# EuroSAT RGB Dataset Profile

## 1. Overview
- **Dataset Name**: EuroSAT (RGB)
- **Total Images**: 27,000
- **Number of Classes**: 10
- **Spatial Resolution**: 64 x 64 pixels (`MAX_IMAGE_SIZE = 64`)
- **Color Channels**: 3 (RGB)
- **Split Seed**: 42

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

## 4. Channel-Wise Pixel Statistics (`[0.0, 1.0]`)
- **Mean (R, G, B)**: `[0.3412, 0.3787, 0.4054]`
- **Standard Deviation (R, G, B)**: `[0.0924, 0.0659, 0.0561]`

## 5. Resolution Audit Output (`outputs/dataset_resolution_audit.txt`)
```text
Dataset: archive/EuroSAT
Total image files: 27000
Resolution counts:
  64 x 64: 27000 images
Maximum resolution: 64 x 64
Unreadable files: 0
PASS: All images satisfy the maximum 64 x 64 requirement.
```
