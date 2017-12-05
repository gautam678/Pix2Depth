

# Dataset
The dataset for this repo can be downloaded [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

Place the downloaded file in the folder data/

For the lazy: run download_nyu_dataset.sh to automatically download the data and save in seperate folders.

# Required Packages
* Keras
* Flask
* opencv
* h5py
* PIL
* numpy

# Running and evaluating

## Configurations
### Structure of the config.py 
`
CONFIG = {
        'development': False,
        'host': [host],
        'port': [port_number],
        'pix2depth':{
                'first_option':'pix2pix',
                'second_option':'CycleGAN',
                'third_option':'CNN',
        },
        'depth2pix':{
                'first_option':'pix2pix',
                'second_option':'CycleGAN',
                'third_option':'MSCNN'
        },
        'portrait':{
                'first_option': 'pix2pix',
                'second_option': 'CycleGAN',
                'third_option': 'CNN'
        }
}

`


## Running the Application

`python app.py`

This will start the python server.


**Example:**

## Output
- The weights are stored in the folder weights/ [main.py requires the path to the weights to load the model]
- The generated images are stored in static/results/ [the images are stored with the name of the model so it's easier to identify results] 

### Additional notes
- Used the following models to train on nyu_depth dataset.
        * [pix2pix](https://github.com/phillipi/pix2pix)
        * [CycleGan](https://github.com/junyanz/CycleGAN)
        * [Multi Scale CNN](https://github.com/alexhagiopol/multiscale-CNN-classifier]

