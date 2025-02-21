# RadiomicsPersona

### Task 1: Train a U-Net Model for MRI Image Reconstruction

**Objective:**

Train a U-Net model to perform 3D MRI image reconstruction by artificially corrupting parts of the healthy MRI images and then having the U-Net reconstruct those missing regions.

![rough_structure](https://github.com/user-attachments/assets/6d0d148a-71b6-4a93-86cf-97ebdadc2ea0)


### Steps:

1. **Input Data**:
    - The input will be a 3D MRI image in `.nii.gz` format.
    - These images should represent healthy tissues.
2. **Corruption (Data Augmentation)**:
    - For each MRI volume, randomly hollow out (remove) a cubic or rectangular region from the 3D image. This can be achieved by zeroing out the voxel values or using a mask to create the "missing" region.
    - The hollowed-out version will be used as the input for the U-Net.
3. **Ground Truth (Target)**:
    - The original (uncorrupted) 3D MRI image will serve as the ground truth.
4. **Training Objective**:
    - The U-Net will be trained to reconstruct the missing region of the MRI image.
    - The loss function could be a voxel-wise mean squared error (MSE) or another reconstruction loss, comparing the generated output with the original healthy tissue.
5. **Model Architecture**:
    - Use a standard 3D U-Net architecture to handle the 3D nature of the MRI images.
    - Input shape: 3D MRI with the hollowed region.
    - Output shape: 3D MRI of the reconstructed region.

### Output:

- A trained U-Net model that can reconstruct missing regions in 3D MRI scans of healthy tissue.

---

### Task 2: Lesion Reconstruction and Classification Model

**Objective:**

Use the trained U-Net model from Task 1 to generate a "healthy" reconstruction of the lesion area and compare it to the original lesion, eventually classifying whether a region is healthy (0) or pathological (1).

### Steps:

1. **Input Data**:
    - Pathological 3D MRI images (`.nii.gz` format).
    - A bounding box that specifies the lesion area within each MRI scan.
2. **Reconstruction of Lesion Area**:
    - Use the U-Net model trained in **Task 1** to reconstruct healthy tissue for the lesion area within the bounding box.
    - The bounding box will be applied to the input MRI, and the U-Net will attempt to reconstruct what a healthy version of that region would look like.
3. **Comparison**:
    - Compare the reconstructed (healthy) version of the lesion area with the original, pathological lesion.
    - The comparison can be done using various similarity metrics such as:
        - **Mean Squared Error (MSE)**: Compare the voxel-wise differences between the lesion region and the U-Net-generated region.
        - **Structural Similarity Index (SSIM)**: Measure structural similarity between the lesion and reconstructed regions.
        - **Cosine Similarity**: Compare the vectorized lesion and reconstructed images.
4. **Classification Task**:
    - Based on the similarity metrics computed in the previous step, classify the lesion as pathological (1) or non-pathological (0).
    - This could be done using a threshold value on the similarity score, or you could train a small classifier (e.g., a simple neural network) that takes the similarity metrics as input and outputs a binary classification.

### Output:

- **Classification Label (0 or 1)**:
    - 0 if the lesion area is reconstructed similarly to the original tissue (indicating non-pathological).
    - 1 if the lesion area is significantly different from the healthy reconstruction (indicating a pathological region).
