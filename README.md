# Getting Started With the Pipeline

This repository contains the pipeline codes for the long-term multiple-brain-region imaging of neuronal and neuromodulatory activities in *Drosophila* brain by the two-photon synthetic aperture microscopy (2pSAM). This pipeline includes Δ*F/F* extraction and data analysis, organized by **Python**.

## Installation

Please install the dependencies using: 

​`pip install -r environment.txt`

## Datasets

* The demo data is currently available on [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/fjq19_mails_tsinghua_edu_cn/EtZeYbE6qfFDpNpT_uv4Mi8BiAGpYAnsJEAz9RsjXmvZdw?e=asuQ30). When using the demo data to test the codes, please change the name of the file folders in the codes according to the storage location of the data.
* The entire dataset with a total size of 5 TB, which includes the extracted neuronal and neuromodulatory traces within the 3D volumes of 10 flies co-labeled by jGCaMP7f (an indicator for calcium) and UAS-rGRAB_ACh-0.5 (an indicator for acetylcholine (ACh)), and 10 flies co-labeled by jGCaMP7f and rGRAB_HTR2C-0.5 (an indicator for serotonin (5-HT)), will be open-sourced after publication. 

## Usage

These codes are tested under Python 3.8.15. The pipeline consists of the following sections:

### 1. Δ*F/F* Trace Extraction

The '**/p1_dff0_extract**' folder consists of codes for calculating Δ*F/F* from the raw traces. 

* Extract Δ*F/F* of each stimulus for each indicator (for the analysis of odor responses):

  `python p1_dff0_extract/c1_dff0_stim.py `

  Due to the large total data size, we extract Δ*F/F* for **each stimulus** rather than the entire experiment. The calculation window for each stimulus is customizable through parameters in the code.

  * **'stim_before_2'**: The start of the windows relative to odor delivery. If the parameter is negative, the windows start after the stimulus.
  * **'sw_len_2'**: Window length for each stimulus.
  * **'time_downsample_2'**: Time downsample rate. For r5-HT responses with a wider pulse width, we expand the window and downsample 2 times to include the response process for each stimulus.

  In order to standardize the storage and usage, the Δ*F/F* traces are named as follows:

  `'dff0_'+str(-stim_before_2[0])+'-'+str(-stim_before_2[0]+sw_len_2[0])+'_down'+str(time_downsample_2[0])+'_C*'`
  *For example, the ΔF/F trace starting from odor delivery with a length of 20 frames without downsampling is named 'dff0_0-20_down1_C\*.npy'*

  The analysis codes below may require distinct Δ*F/F*, as specified in the file loading section of each code.<br/>
- Extract Δ*F/F* traces of a segment preceding the first stimulus for each indicator (for the analysis of the resting state):

  `python p1_dff0_extract/c2_extract_start_pre.py `

### 2. Basic Analysis

The '**/p2_basic_analysis**' folder consists of codes for basic analysis of the olfactory responses and reproducing the related figures in the paper, including trace plots, maps of several measurements, and calculation of the dynamic properties of the responses. The figure indices are shown in the file names.

### 3. Representation Analysis

The '**/p3_representation_analysis**' folder consists of codes for odor identity representation analysis and reproducing the related figures in the paper.

* The main codes for odor identity representation analysis of each fly<br/>
  The '**/1-main-processing**' folder:
  * 'p1-voxel-level-whole-brain.ipynb': Voxel-level multiple-brain-region odor identity representation
  * 'p2-voxel-level-each-region.ipynb': Voxel-level odor identity representation in each brain region
  * 'p3-region-level-whole-brain.ipynb':Region-level multiple-brain-region odor identity representation
  * 'p4-accuracy-map.ipynb': Generate maps of decoding accuracy across the FOV
  <br/>

* Statistics and analyses of the results of multiple flies
  <br/>

  The '**/2-supfigure4_p1_to_p3_thresh**' folder: Set the thresholds of the PCA results for p1-p3
  The '**/3-classification_compare**' folder:  
  * '1_\*.ipynb' to '7_2_\*.ipynb': Compare the odor identity decoding accuracies of different situations, the figure indices shown in the file names
  * '8_1_manifold_plot_single.ipynb' and '8_2_manifold_plot_batch.ipynb': Plot the manifolds of a single file and all files of the selected flies, respectively
  <br/>

* Low-dimensional manifold analysis
  <br/>

  The '**/4-manifold-statistics-final**' folder:  
  * '1_cv_merge.ipynb': Align and combine the manifolds of each fold for the following analyses
  * '2_\*.ipynb' to '8_\*.ipynb': Analyze the temporal changes of the manifolds
  * '9_\*.ipynb' to '10_\*.ipynb': Compare the manifolds of each channel and odor identity
  <br/>

* Motion analysis<br/>
  The '**/5-video-analysis**' folder: 
  * The '**/video_processing**' folder: MATLAB codes for motion extraction from the videos of the fly abdomens
  * '0-video-analysis-single-fly.ipynb': Motion analysis for each fly
  * '1_\*.ipynb' to '6_\*.ipynb': Codes for figures in Fig. S6,  the figure indices shown in the file names
  <br/>

* Other analyses<br/>
  The '**/6-others**' folder: 
  * Some other figure plotting codes with the figure indices shown in the file names

### 4. Network Analysis

The '**/p4_network_analysis**' folder consists of codes for network analysis and reproducing the related figures in the paper.

* Generate functional connectivity matrices and networks
  <br/>

  * The '**/01_Generate_G7f_matrices_and_networks**' folder: Generate functional connectivity matrices and networks for G7f both at the regional and neuronal level

  * The '**/02_Generate_ACh_matrices_and_networks**' folder: Generate functional connectivity matrices and networks for ACh both at the regional and neuronal level

  * The '**/03_Generate_5HT_matrices_and_networks**' folder: Generate functional connectivity matrices and networks for 5-HT both at the regional and neuronal level
  <br/>

* Brain-region-level analysis
  <br/>

  * '**/04_Brain-region-level_analysis/Brain-region-level_data_analysis_RS.ipynb**': Generate average matrices and networks of 10 or 20 flies in the resting state (RS)

  * '**/04_Brain-region-level_analysis/Brain-region-level_data_analysis_Stim.ipynb**': Generate average matrices and networks of 10 or 20 flies during odor stimulation (Stim)

  * '**/04_Brain-region-level_analysis/Brain-region-level_results_graphing.ipynb**': Reproduce the results and figures of brain-region-level analysis
  <br/>
  
- Voxel-level analysis
  <br/>

  - '**/05_Single_region_analysis/Single_region_ACh_results_graphing.ipynb**': Reproduce ACh analysis results and figures at the voxel level within brain regions

  - '**/05_Single_region_analysis/Single_region_5HT_results_graphing.ipynb**': Reproduce 5-HT analysis results and figures at the voxel level within brain regions
  <br/>

- Temporal analysis
  <br/>

  * '**/06_Temporal_analysis/Temporal_data_analysis.ipynb**': Generate average matrices and networks of 10 or 20 flies in different time periods

  * '**/06_Temporal_analysis/Temporal_results_graphing.ipynb**': Reproduce the results and figures of temporal analysis


## References

[1] Z. Zhao, Y. Zhou, B. Liu, J. He, J. Zhao, Y. Cai, J. Fan, X. Li, Z. Wang, Z. Lu, J. Wu, H. Qi, Q. Dai, [Two-photon synthetic aperture microscopy for minimally invasive fast 3D imaging of native subcellular behaviors in deep tissue](https://www.cell.com/cell/fulltext/S0092-8674(23)00412-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867423004129%3Fshowall%3Dtrue). *Cell* **186**, 2475-2491.e22 (2023).

[2] X. Li, G. Zhang, J. Wu, Y. Zhang, Z. Zhao, X. Lin, H. Qiao, H. Xie, H. Wang, L. Fang, Q. Dai, [Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising](https://www.nature.com/articles/s41592-021-01225-0). *Nat Methods* **18**, 1395–1400 (2021).

[3] X. Li, Y. Li, Y. Zhou, J. Wu, Z. Zhao, J. Fan, F. Deng, Z. Wu, G. Xiao, J. He, Y. Zhang, G. Zhang, X. Hu, X. Chen, Y. Zhang, H. Qiao, H. Xie, Y. Li, H. Wang, L. Fang, Q. Dai, [Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit](https://www.nature.com/articles/s41587-022-01450-8). *Nat Biotechnol* **41**, 282–292 (2023).

[4] X. Li, X. Hu, X. Chen, J. Fan, Z. Zhao, J. Wu, H. Wang, Q. Dai, [Spatial redundancy transformer for self-supervised fluorescence image denoising](https://www.nature.com/articles/s43588-023-00568-2). *Nat Comput Sci* **3**, 1067–1080 (2023).

## Citation

If you use the resources in this repository, please cite the following paper:

Prominent involvement of acetylcholine in shaping stable olfactory representation across the *Drosophila* brain.
