# Neural Network Verification and Repair for Probabilistic Specifications

## Setup

 - Install Conda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
 - Configure the pytorch installation by editing the `environment.yaml` file.
   Make sure the CUDA version matches your installation or install pytorch without
   GPU support if your system doesn't have a GPU.
   You can find more information on installing pytorch 
   [here](https://pytorch.org/get-started/previous-versions/#v1121)
 - Create a new conda environment using
   ```shell
   conda env create -n YOUR_PREFERRED_ENVIROMENT_NAME -f environment.yaml
   ```
 - Install `auto_LiRPA` using
   ```shell
   git sumbodule update --init  # pull auto_LiRPA if not already present
   conda activate YOUR_PREFERRED_ENVIRONENT_NAME
   cd auto_LiRPA
   python setup.py install
   ```
 Now you are ready to run the experiments in this repository.
