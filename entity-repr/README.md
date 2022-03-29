# expert-entity-repr

## Start running some pilot experiments
* `source setup_env.sh`. Make sure the conda environment is installed to a valid path on your machine.
* `bash run_properties_embedding.sh 1 "0"`. This script will process the SciREX dataset first, and cache it to a folder under `logging/`. Next time when you run this script, set `--if_create_tokenized_data_file` to `"no"` to use the cached data. `1` in the command means you want to use 1 GPU, and `"0"` in the command means the CUDA visible devices. Alternatively, if you have 4 GPUs on the machine, you can use `bash run_properties_embedding.sh 4 "0,1,2,3"`. After processing the data, this script will try to optimize the alignment and uniformity properties, defined by the embedding representation method. We might want to tune `align_weight_over_uniform` in the script. Make sure that other paths in the script is compatible with your machine.
* `bash run_properties_gradient.sh 1 "0"` should work similarly, but with properties defined by the gradient representation method.
* Check `my_tune_lm_properties.py` for more details.
