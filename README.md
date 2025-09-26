
# MAIL: Multi-Modal Interactive Agent Layer for Few-Shot Universal Cross-Domain Retrieval and Beyond [NeurIPS 2025]
> Official github repository for MAIL: **Multi-Modal Interactive Agent Layer for Few-Shot Universal Cross-Domain Retrieval and Beyond**

> The code of applying MAIL to few-shot classification can be found at [MAIL-for-classification](https://github.com/pSGAme/MAIL-for-Classification)

## Requirements

- **Ubuntu**  22.04.4
- **RTX4090D-24GB or better**
- **CUDA** 11.3
- **python** 3.10.4
- **torch** 1.12.1 py3.10_cuda11.3_cudnn8.3.2_0
- **torchvision** 0.13.1 py310_cu113

## Setups

1. download the zip file.

2. Install dependencies：

```bash
cd ./MAIL
conda env create -f MAIL.yaml
conda activate MAIL
```

## Data Preparation

1. Download DomainNet using scripts in `MAIL/src/data/downloads`.

   ``` bash
   cd ./src/data/downloads
   bash download_domainnet.sh
   ```
2. For Sketchy and TU-berlin, please refer to  [ProS](https://github.com/kaipengfang/ProS).

3. The directory is expected to be in the structure below:

   ```python
   ├── DomainNet
   │   ├── clipart # images from clipart domain
   │   ├── clipart_test.txt # class names for testing
   │   ├── clipart_train.txt # class names for training
   │   ├── down.sh
   │   ├── infograph
   │   ├── infograph_test.txt
   │   ├── infograph_train.txt
   │   ├── painting
   │   ├── painting_test.txt
   │   ├── painting_train.txt
   │   ├── quickdraw
   │   ├── quickdraw_test.txt
   │   ├── quickdraw_train.txt
   │   ├── real
   │   ├── real_test.txt
   │   ├── real_train.txt
   │   ├── sketch
   │   ├── sketch_test.txt
   │   └── sketch_train.txt
   ├── Sketchy
   │   ├── extended_photo
   │   ├── photo
   │   ├── sketch
   │   └── zeroshot1
   └── TUBerlin
       ├── images
       └── sketches
   ```

## Experiments
The algorithms are in ./src/algos

Be sure to modifiy the `data_path` and `code_path` in each main.py
## baselines:

### 1: MaPLe+IVLP+VPT+VPT-D:

```bash
cd ./src/algos/1_PromptFamily
sh run.sh
```

### 2: MMA:

```bash
cd ./src/algos/2_MMA
sh run.sh
```
### 3: IVLA:

```bash
cd ./src/algos/3_IVLA
sh run.sh
```


### 4: LoRA:

```bash
cd ./src/algos/4_LoRA
sh run.sh
```

### 5: BitFit:

```bash
cd ./src/algos/5_BitFit
sh run.sh
```

### 6: ProS:

```bash
cd ./src/algos/6_ProS
sh run.sh
```


### MAIL (Ours):

```bash
cd ./src/algos/MAIL
sh run.sh
```


## Acknowledgements

Our code implementation is based on [CoOp](https://github.com/KaiyangZhou/CoOp), [MaPle](https://github.com/muzairkhattak/multimodal-prompt-learning), 
[MMA](https://github.com/muzairkhattak/multimodal-prompt-learning), [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA), and [ProS](https://github.com/kaipengfang/ProS).
We thank all the authors for releasing their code.




