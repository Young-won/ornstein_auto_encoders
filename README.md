# Learning from Nested Data with Ornstein Auto-Encoders

## Environment

### (recommended) Using Docker
Install ddo (https://docs.docker.com/get-docker/)
More information for docker can be found in [Docker documentation](https://docs.docker.com/storage/bind-mounts/)

#### Build the docker container
```
docker build -t oae_docker .
```

#### Run the docker container
The docker, by default, start in the current directory as /workdir. If you need to access your local directories, you need to be mounted in the docker.

To mount a directory, use the -v <source_dir>:<mount_dir> option.
More information for mounting directories can be found in [Docker documentation](https://docs.docker.com/storage/bind-mounts/)

```
docker run -it --rm -v $PWD:/workdir -w /workdir oae_docker bash
```

#### Run the jupyter notebook with the dockr container
```
export notebook_port=9999
docker run -it --rm -p $notebook_port:$notebook_port -v $PWD:/workdir -w /workdir oae_docker bash -c "jupyter notebook --port=$notebook_port --no-browser --ip=0.0.0.0 --allow-root"
```

### Pip environments

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Set Data

### VGGFace2
You can download VGGFace2 data here:

- [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) 

Set all information from VGGFace2 dataset under the directory "data/vggface2/". It should have the files:
- "train_list.txt"
- "test_list.txt"
- "identity_meta.csv"

and the directories of images:
- "train/"
- "test/"
- "bb_landmark/"


To preprocess the VGGFace2 data, run this command:

```preprocessing
python3.6 vggface2_preprocessing.py --log_info=configurations/log_info.yaml --data_dir=./data/vggface2
```
You should set the directory path regarding the VGGFace2 dataset to the argument `--data_dir`.
Note that this command will take several hours to process with a single NVIDIA TITAN X.

## Training

### Imbalanced MNIST

To train the product-space OAE model in the paper, run this command:

```train
python3.6 mnist_training.py --log_info=configurations/log_info.yaml --path_info=configurations/mnist/psoae_path_info.cfg --network_info=configurations/mnist/psoae_network_info.cfg
```

The trained model will be saved at the directory `mnist_experiments/mnist_imbalance_psoae`.


### VGGFace2

To train the product-space OAE model in the paper, run this command:

```train
python3.6 vggface2_training.py --log_info=configurations/log_info.yaml --path_info=configurations/vggface2/psoae_path_info.cfg --network_identity_info=configurations/vggface2/psoae_network_identity_info.cfg --network_within_unit_info=configurations/vggface2/psoae_network_within_unit_info.cfg --network_total_info=configurations/vggface2/psoae_network_total_info.cfg
```

The trained model will be saved at the directory `vggface2_experiments`.


## Evaluation

### Imbalanced MNIST

You can use the pre-trained models on imbalanced MNIST in the directory "mnist_pretraind/".

To evaluate the pre-trained model on imbalanced MNIST, see the notebooks:
    - "mnist_numerics.ipynb" for calculating the performance measures
    - "mnist_figures.ipynb" for generating the figures


### VGGFace2

The pre-trained models on VGGFace2 was omitted because of the size. The download link is as follow:
https://drive.google.com/file/d/1035W4rNhacXkCrYt9YSKt7nf4WiTqVIU/view?usp=sharing

To evaluate the pre-trained model on VGGFace2, unzip the downloaded model to specipic directory (ex. vggfae2_pretrained/vggface2_psoae) and run this command:

```eval
python3.6 vggface2_evaluate.py --log_info=configurations/log_info.yaml --model_path=vggface2_pretrained/vggface2_psoae --model_aka=PSOAE --path_info=configurations/vggface2/psoae_path_info.cfg --network_info=configurations/vggface2/psoae_network_total_info.cfg
```

Note that you have to set the directory have the pre-trained model in the argument ` --model_path`.

To generate the images with the pre-train model on VGGFace2, see the notebook "vgg2_figures.ipynb"


## Results

Our model (product-space OAE) achieves the following performance on :

### Imbalanced MNIST

|                      |        Accuracy       |   One-shot accuracy   |          SSIM         |       Sharpness       |
|----------------------|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| WAE                  |                     - |                     - |                     - | \(0.047 ($\pm$ 0.003)\) |
| CAAE                 | \(0.860 ($\pm$ 0.008)\) |                     - | \(0.244 ($\pm$ 0.020)\) | \(0.041 ($\pm$ 0.007)\) |
| RIOAE | \(0.919 ($\pm$ 0.033)\) | \(0.873 ($\pm$ 0.059)\) | \(0.292 ($\pm$ 0.017)\) | \(0.025 ($\pm$ 0.004)\) |
| __PSOAE__    | \(0.939 ($\pm$ 0.019)\) | \(0.878 ($\pm$ 0.029)\) | \(0.263 ($\pm$ 0.008)\) | \(0.032 ($\pm$ 0.008)\) |
| Testset              | \(0.994 ($\pm$ 0.002)\) |                     - | \(0.229 ($\pm$ 0.009)\) | \(0.075 ($\pm$ 0.004)\) |


### VGGFace2

|                      | Identities used in training |                         |                        | Identities not used in training |                         |                       |
|----------------------|:---------------------------:|:-----------------------:|:----------------------:|:-------------------------------:|:-----------------------:|:---------------------:|
|                      |              IS             |           FID           |        Sharpness       |                IS               |           FID           |       Sharpness       |
| WAE                  |            \(-\)            |            -            |            -           |     \( 2.125 ($\pm$ 0.016) \)     | \(106.250 ($\pm$ 3.024)\) | \(0.001 ($\pm$ 0.000)\) |
| CAAE                 |    \(2.029 ($\pm$ 0.010)\)    | \(115.767 ($\pm$ 2.796)\) |  \(0.001 ($\pm$ 0.000)\) |                -                |            -            |           -           |
| RIOAE |    \(2.068 ($\pm$ 0.011)\)    | \(107.961 ($\pm$ 2.371)\) | \( 0.001 ($\pm$ 0.000)\) |      \(2.067 ($\pm$ 0.020)\)      | \(102.476 ($\pm$ 3.363)\) | \(0.001 ($\pm$ 0.000)\) |
| __PSOAE__    |    \(2.146 ($\pm$ 0.104)\)    |  \(98.525 ($\pm$ 2.487)\) |  \(0.001 ($\pm$ 0.000)\) |     \( 2.125 ($\pm$ 0.118) \)     |  \(94.287 ($\pm$ 2.323)\) | \(0.001 ($\pm$ 0.000)\) |
| Testset              |    \(3.883 ($\pm$ 0.146)\)    |            -            |  \(0.004 ($\pm$ 0.001)\) |      \(3.807 ($\pm$ 0.164)\)      |            -            | \(0.003 ($\pm$ 0.001)\) |
