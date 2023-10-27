# BirdSAT: Cross-View Contrastive Masked Autoencoders for Bird Species Classification and Mapping

## 🦢 Dataset Released: Cross-View iNAT Birds 2021
This cross-view birds species dataset consists
of paired ground-level bird images and satellite images, along with meta-information associated with the iNaturalist-2021 dataset.

![CiNAT-Birds-2021](imgs/data5%20(2).png)

#### Computer Vision Tasks
1. Fine-Grained image classification
2. Satellite-to-bird image retrieval
3. Bird-to-satellite image retrieval
4. Geolocalization of Bird Images

An example of task 3 is shown below:

![Retrieval](imgs/ret_ex.png)

## 👨‍💻 Getting Started 

#### Installing Required Packages
There are two options to setup your environment to be able to run all the functions in the repository:
1. Using Dockerfile provided in the repository to create a docker image with all required packages:
    ```bash
    docker build -t <your-docker-hub-id>/birdsat .
    ```
2. Creating conda Environment with all required packages:
    ```bash
    conda create -n birdsat python=3.10 && \
    conda activate birdsat && \
    pip install requirements.txt
    ```

Additionally, we have hosted a pre-built docker image on docker hub with tag `srikumar26/birdsat:latest` for use.

#### Data Preparation
Please refer to `./data/README.md` on instructions for downloading and preparing data.

## 🔥 Training Models
1. Setup all the parameters of interest inside `config.py` before launching the training script.
2. Run pre-training by calling:
    ```bash
    python pretrain.py
    ```
3. Run fine-tuning by calling:
    ```bash
    python finetune.py
    ```

## ❄️ Pretrained Models
Download pretrained models from the given links below:

|Model Type|Download Url|
|----------|--------|
|CVE-MAE|[Link]()|
|CVE-MAE-Meta| [Link]()|
|CVM-MAE| [Link]()|
|CVM-MAE-Meta| [Link]()|


## 📑 Citation

```bibtex
@inproceedings{sastry2024birdsat,
  title={BirdSAT: Cross-View Contrastive Masked Autoencoders for Bird Species Classification and Mapping},
  author={Srikumar, Sastry and Subash, Khanal and Huang, Di and Aayush, Dhakal and Nathan, Jacobs},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
```