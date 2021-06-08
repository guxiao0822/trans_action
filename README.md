# TransAction
This is the implementation of ICL-SJTU submission to Epic-Kitchens Action Anticipation Challenge 2021. 

## Training
#### data preparation
* download pre-extracted multi-modal features provided by [RULSTM](https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek100_full.sh)
* download the [annotations](https://github.com/fpv-iplab/rulstm/tree/master/RULSTM/data/ek100) 

#### training 
* run the following script replacing Data_DIR with your own data folder path.

`python main.py --root_dir Data_DIR`

## Acknowledgement
* RULSTM `https://github.com/fpv-iplab/rulstm`;
* Equalization loss `https://github.com/tztztztztz/eql.detectron2`.

