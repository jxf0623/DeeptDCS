# DeeptDCS: Deep Learning-Based Estimation of Currents Induced During Transcranial Direct Current Stimulation

This is an implementation of a deep learning-based transcranial direct current stimulation (tDCS) emulator named DeeptDCS (https://arxiv.org/abs/2205.01858) on Python 3, Keras, and TensorFlow. 

The emulator leverages Attention U-net taking the volume conductor models (VCMs) of human head tissues as inputs and outputting the three-dimensional current density distribution across the entire head. 

U-net and its four variants are implemented and their performance are compared. 

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing our paper (bibtex below).

<p align="center">
<img src='./DeeptDCS_workflow.png' align="center" width="600">
</p>

## Requirements
- Python 3.6
- TensorFlow 2.2.0
- Keras 2.3.0-tf
- Other common libraries

## Data structure
Data is structured as follows:
```markdown
├── dir(subject ID 1)
│   ├── dir(Montage Position1)
|	|	├── dir(1)
|	|	|	├── dir(field\_cond.mat)
|	|	├── dir(2)
|	|	|	├── dir(field\_cond.mat)
			...
│   ├── dir(Montage Position 2)
			...
│   ├── dir(Montage Position 3)
			...
│   ├── dir(Montage Position 4)
			...
│   ├── dir(Montage Position 5)
			...
├── dir(subject ID 2)
			...
└── dir(subject ID n)
```
An example dataset can be downloaded from DeeptDCS_data_samples.

## Model Zoo
This directory contains the pre-trained models presented in our DeeptDCS paper. 
To use the pre-trained model, please rename the model as 'DeeptDCS.hdf5' and copy it to folder './saved_model/'.
1. Models pre-trained on 59,000 samples constructed from 59 MRIs and five 5*5cm<sup>2 </sup> square montage positions.

- Standard U-net --------- ```Unet_908137[9].hdf5```
- Attention U-net -------- ```AttnUnet_908136[11].hdf5```
- Res U-net --------------- ```ResUnet_913918[5].hdf5```
- Attention Res U-net---- ```AttnResUnet_913915[5].hdf5```
- MSResU-net ------------ ```MSResUnet_913917[8].hdf5```


3. Attention U-net models fine-tuned on
- three new montage positions --------------------------------```AttnUnet_new_positions_18194801[5].hdf5```
- new montage shapes (5 cm diameter circular montage) --------```AttnUnet_ellp5050_19000238[3].hdf5```
- new  montage size (4*4cm<sup>2 </sup> square montage) ------```AttnUnet_rect4040_19000241[4].hdf5```



## Training and Testing
***Neural network selection***

U-net and its four variations are implemented. The default network is Attention U-net. To change the network, please search ```self.model = self.attn_unet_3d``` in model.py and replace the command by one of the following:
```
self.model = self.unet_3d
self.model = self.MSResUnet_3d
self.model = self.ResUnet_3d
self.model = self.attn_unet_3d
self.model = self.attn_ResUnet_3d
```

1. Train and test a new model from scratch
- Clear or delete directory './saved_mode'
```
python train DeeptDCS.py --data_path=/path/to/dataset/
```
2. Test the well-trained model
- Copy the model to  ```'./saved_mode'``` and rename it as 'DeeptDCS.hdf5'
- In DeeptDCS.py, set ```args.epochs = 0```
```
python train DeeptDCS.py --data_path=/path/to/dataset/
```
3. Transfer learning for non-trained montage configurations
- Copy the pre-trained model to  './saved_mode' and rename it as 'DeeptDCS.hdf5'
- In dataloader.py, change ```self.subjectIDs```, ```self.electrode_positions```, ```self.data_size_1position_1subject``` according to the new dataset.
```
python train DeeptDCS.py --data_path=/path/to/transfer_learning_dataset/
```
4. Train and test on customdataset
- Change ```dataloader.py``` to load data according to the costume dataset sturcutre.

## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

# Acknowledgement
gh repo clone IntelAI/unet

# TO DO
写repo includes
改citation
DeeptDCS_data_samples加超链接
改licence
