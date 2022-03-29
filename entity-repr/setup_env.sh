# setup conda environment on a new machine
conda create --name lmi python=3.8
# conda create --prefix lmi python=3.8

# activate the new environment
conda activate lmi
# conda activate lmi

# after environment activation
pip install jupyterlab numpy scipy tqdm seaborn==0.11.2

# install pytorch (note the cuda driver version on the new machine)
# conda install pytorch cudatoolkit=10.1 -c pytorch-lts # pytorch version 1.8.2
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia # newer cuda version

# install huggingface transformers library
pip install transformers==4.12.3 # version 4.10.0; 4.11.3; 4.12.3

# additional packages
pip install datasets==1.15.1 # version 1.11.0; 1.14.0; 1.15.1
pip install accelerate==0.5.1 # version 0.4.0; 0.5.1

# dependencies for openprompt
pip install yacs scikit-learn dill sentencepiece==0.1.96 tensorboardX

# for jupyter lab
pip install ipywidgets termcolor statsmodels
