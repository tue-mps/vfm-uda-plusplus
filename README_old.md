# VFM-UDA++: Improving Network Architectures and Data Strategies for Unsupervised Domain Adaptive Semantic Segmentation 
**Authors:** Brunó B. Englert, Gijs Dubbelman  \
**Affiliation:** Eindhoven University of Technology \
**Paper:** [arXiv](https://arxiv.org/abs/2503.10685)   
**Code**: [GitHub](https://github.com/tue-mps/vfm-uda-plusplus) 


## Abstract
Unsupervised Domain Adaptation (UDA) has shown remarkably strong generalization from a labeled source domain to an unlabeled target domain while requiring relatively little data. At the same time, large-scale pretraining without labels of so-called Vision Foundation Models (VFMs), has also significantly improved downstream generalization. This motivates us to research how UDA can best utilize the benefits of VFMs. The earlier work of VFM-UDA showed that beyond state-of-the-art (SotA) results can be obtained by replacing non-VFM with VFM encoders in SotA UDA methods. In this work, we take it one step further and improve on the UDA architecture and data strategy themselves. We observe that VFM-UDA, the current SotA UDA method, does not use multi-scale inductive biases or feature distillation losses, while it is known that these can improve generalization. We address both limitations in VFM-UDA++ and obtain beyond SotA generalization on standard UDA benchmarks of up to +5.3 mIoU. Inspired by work on VFM fine-tuning, such as Rein, we also explore the benefits of adding more easy-to-generate synthetic source data with easy-to-obtain unlabeled target data and realize a +6.6 mIoU over the current SotA. The improvements of VFM-UDA++ are most significant for smaller models, however, we show that for larger models, the obtained generalization is only 2.8 mIoU from that of fully-supervised learning with all target labels. Based on these strong results, we provide essential insights to help researchers and practitioners advance UDA.


## Installation 
1. **Create a Weights & Biases (W&B) account.**
   - The metrics during training are visualized with W&B: https://wandb.ai 

2. **Environment setup.**
     ```bash 
    conda create -n vfmudapp python=3.10 && conda activate vfmudapp
    ```

3. **Install required packages.**
    ```bash
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1
    python3 -m pip install -r requirements.txt
    conda install nvidia/label/cuda-12.4.0::cuda
    ```
   
4. **Compile deformable attention.**
    ```bash
    cd ops
    python3 setup.py build install
    ```

## Data preparation

- **GTA V**: [Download 1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip) | [Download 2](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_images.zip) | [Download 3](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_images.zip) | [Download 4](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_images.zip) | [Download 5](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_images.zip) | [Download 6](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_images.zip) | [Download 7](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_images.zip) | [Download 8](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_images.zip) | [Download 9](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip) | [Download 10](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip) | [Download 11](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) | [Download 12](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip) | [Download 13](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip) | [Download 14](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip) | [Download 15](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip) | [Download 16](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip) | [Download 17](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip) | [Download 18](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip) | [Download 19](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip) | [Download 20](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip)
- **Synthia**:  [Download 1](http://synthia-dataset.net/download/808/) 
- **Synscapes**:  [Download 1](https://synscapes.on.liu.se/download.html)
    - *Note: this step requires 700GB of free storage space*
    - ```bash
      tar -xf synscapes.tar
      zip -r -0 synscapes.zip synscapes/
      rm -rf synscapes.tar
      rm -rf synscapes
      ```
- **Cityscapes**: [Download 1](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Download 2](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
- **Mapillary**: [Download 1](https://www.mapillary.com/dataset/vistas)
- **ACDC**:    [Download 1](https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip) |  [Download 2](https://acdc.vision.ee.ethz.ch/gt_trainval.zip) 
- **DarkZurich**:  [Download 1](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip) | [Download 2](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip) 
- **BDD100K**: [Download 1](https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_images_10k.zip) |  [Download 2](https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_sem_seg_labels_trainval.zip)
- **WildDash**: [Download 1](https://wilddash.cc/download/wd_public_02.zip) (Download the "old WD2 beta", not the new "Public GT Package")
  - For WilDdash, an extra step is needed to create the train/val split. After the "wd_public_02.zip" is downloaded, place the files from the "wilddash_trainval_split" in the same direcetory as the zip file. After that, run:
    ```bash 
    chmod +x create_wilddash_ds.sh
    ./create_wilddash_ds.sh
    ```
    This creates a new zip files, which should be used during training.
    
All the zipped data should be placed under one directory. No unzipping is required.

## Usage

### Training

To train the VFM-UDA++ large model from scratch, run:
   ```bash
   python3 main.py fit -c vfmudaplusplus_large_gta2city.yaml --root /data  --trainer.devices "[0, 1, 2, 3]"
   ```
   (replace ```/data``` with the folder where you stored the datasets)

*Note: that there are small variations in performance between training runs, due to the stochasticity in the process, particularly for UDA techniques. Therefore, results may differ slightly depending on the random seed.*


### Evaluating
To evaluate a pre-trained VFM-UDA++ model, run:

```bash
python3 main.py validate -c vfmudaplusplus_large_gta2city.yaml --root /data  --trainer.devices "[0, 1, 2, 3]" --model.network.pretrained_adapter_path "/path/to/checkpoint.ckpt"
```
(replace ```/data``` with the folder where you stored the datasets)



## Model Zoo
### Main results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Architecture</th>
<th valign="center">Dataset Scenario</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">Cityscapes (miou)</th>
<th valign="bottom">WildDash2 (miou)</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-L + ViT Adapter + BasicPyramid</td>
<td align="center">GTA5 to City</td>
<td align="center">DINOv2</td>
<td align="center">0.79849</td>
<td align="center">0.68965</td>
<td align="center">xkp2etl0</td>
</tr>

<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-L + ViT Adapter + BasicPyramid</td>
<td align="center">All Synth to All Real</td>
<td align="center">DINOv2</td>
<td align="center">0.82195</td>
<td align="center">0.71266</td>
<td align="center">1wqcuk69</td>
</tr>

<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-L + ViT Adapter + BasicPyramid</td>
<td align="center">Synthia to Cityscapes</td>
<td align="center">DINOv2</td>
<td align="center">0.697098125</td>
<td align="center">0.560749375</td>
<td align="center">ehxxeooc</td>
</tr>

<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-L + ViT Adapter + BasicPyramid</td>
<td align="center">Cityscapes to Darkzurich</td>
<td align="center">DINOv2</td>
<td align="center">68.69</td>
<td align="center">70.315</td>
<td align="center">k2ha39fy</td>
</tr>
</tbody></table>

*Note: these models are re-trained, so the results differ slightly from those reported in the paper.*


### ImageNet1k pretrained ViT-Adapters


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Architecture</th>
<th valign="bottom">Dataset Scenario</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->


<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-S + ViT Adapter + BasicPyramid</td>
<td align="center">IN1k</td>
<td align="center">DINOv2</td>
<td align="center">adkoomia</td>
</tr>
<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-B + ViT Adapter + BasicPyramid</td>
<td align="center">IN1k</td>
<td align="center">DINOv2</td>
<td align="center">e4913zt5/kodn7tt9</td>
</tr>

<tr><td align="left">VFM-UDA++</td>
<td align="center">ViT-L + ViT Adapter + BasicPyramid</td>
<td align="center">IN1k</td>
<td align="center">DINOv2</td>
<td align="center">7ndph8rf</td>
</tr>

</tbody></table>

## Citation
```
@inproceedings{englert2025vfmudaplutplus,
  author={{Englert, Brunó B.} and {Dubbelman, Gijs}},
  title={VFM-UDA++: Improving Network Architectures and Data Strategies for Unsupervised Domain Adaptive Semantic Segmentation},
  eprint={2503.10685},
  url={https://arxiv.org/abs/2503.10685}, 
  year={2025},
}
```

## Acknowledgement
We use some code from:
 * DINOv2 (https://github.com/facebookresearch/dinov2): Apache-2.0 License
 * Masked Image Consistency for Context-Enhanced Domain Adaptation (https://github.com/lhoyer/MIC): Copyright (c) 2022 ETH Zurich, Lukas Hoyer, Apache-2.0 License
 * SegFormer (https://github.com/NVlabs/SegFormer): Copyright (c) 2021, NVIDIA Corporation, NVIDIA Source Code License
 * DACS (https://github.com/vikolss/DACS): Copyright (c) 2020, vikolss, MIT License 
 * MMCV (https://github.com/open-mmlab/mmcv): Copyright (c) OpenMMLab, Apache-2.0 License
