# RadiomicsPersona

### Patient-specific radiomic feature search with reconstructed healthy persona of knee MR images

### Overview
Classical radiomic features (e.g., entropy, energy) have been designed to describe image appearance and intensity patterns. These features are directly interpretable and readily understood by radiologists. Compared with end-to-end deep learning (DL) models, lower dimensional parametric models that use such radiomic features offer enhanced interpretability but lower comparative performance in clinical tasks. In this study, we propose an approach where a standard logistic regression model performance is substantially improved by learning to select radiomic features, for individual patients, from a pool of candidate features. This approach has potentials to maintain the interpreatability of such approaches while offering similar performance to DL. In addition, we expand the feature pool by generating a patient-specific “healthy persona” via mask-inpainting using a denoising diffusion model trained on healthy subjects. Such a non-diseased baseline feature set allows not only further opportunity in novel feature discovery but also improved condition classification.
We demonstrate our method on clinical tasks such as classifying anterior cruciate ligament (ACL) tears, meniscal tears, and other abnormalities. Experimental results demonstrate that our approach achieved superior performance compared to state-of-the-art DL approaches while offering added interpretability through the use of radiomic features extracted from images and supplemented by generating healthy personas. These findings highlight the potential of generative models in augmenting radiomic analysis for more robust and interpretable clinical decision-making. 

## **Prerequisites**
Before running the scripts in this project, ensure you have the following dependencies installed and properly configured.

### **Dataset**
We utilize the [MRNet dataset](https://stanfordmlgroup.github.io/competitions/mrnet/) for our experiments. Ensure all images are renamed according to the following convention:

- **Coronal View**: `CORO_000_0001.nii.gz`
- **Sagittal View**: `SAGI_000_0001.nii.gz`
- **Axial View**: `AXIA_000_0001.nii.gz`

After image registration, the naming convention follows this pattern: `res_CORO_000_0001.nii.gz`. The dataset processing script ([dataset.py](dataset.py)) aligns with this naming convention.

### **Image Registration**
To ensure accurate alignment and transformation of images, we use **NiftyReg**, a software package for efficient medical image registration.

#### **Installing NiftyReg**
1. **Download NiftyReg**: Visit [NiftyReg Installation Guide](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install) and download the latest version.
2. **Compile NiftyReg**: Follow the provided compilation instructions to build the software. Ensure that all necessary dependencies are installed and configured properly.

---

For further details on running experiments, data preprocessing, and model training, refer to the respective scripts and documentation provided in the repository.
