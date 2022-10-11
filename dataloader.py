#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#


import os
import numpy as np
from argparser import args

import gzip, pickle, pickletools
from scipy.io import loadmat


TRAIN_TESTVAL_SEED = 816

args.keras_api = False

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras/TensorFlow

    This uses the Keras Sequence which is a better data pipeline.
    It will allow for multiple data generator processes and
    batch pre-fetching.

    If you have a different type of dataset, you'll just need to
    change the loading code in self.__data_generation to return
    the correct image and label.

    """

    def __call__(self, *args, **kwargs):
        return self.__getitem__

    def __init__(self,
                 setType,  # ["train", "validate", "test"]
                 data_path,  # File path for data
                 batch_size=8,  # batch size
                 dim=(144, 144, 144),  # Dimension of images/masks
                 n_in_channels=1,  # Number of channels in image
                 n_out_channels=1,  # Number of channels in mask
                 shuffle=True,  # Shuffle list after each epoch
                 seed=816,  # Seed for random number generator
                 varification_test = False):
        """
        Initialization
        """
        self.data_path = data_path

        if setType not in ["train", "test", "validate"]:
            print("Dataloader error.  You forgot to specify train, test, or validate.")

        self.setType = setType
        self.dim = dim
        self.batch_size = batch_size

        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.seed = seed
        self.varification_test = varification_test
        self.subjectIDs_filtered_85=[
            'mmnd_sub-13_ses-mri_anat_sub-13_ses-mri_run-2_echo-1_FLASH',
            'mmnd_sub-14_ses-mri_anat_sub-14_ses-mri_run-2_echo-1_FLASH',
            'mmnd_sub-15_ses-mri_anat_sub-15_ses-mri_run-1_echo-5_FLASH',
            'OAS2_0012_MR1_4_123127264_mpr-4_nifti','OAS2_0013_MR1_2_123127471_mpr-2_nifti',
            'OAS2_0018_MR1_3_123128200_mpr-3_nifti','OAS2_0032_MR2_2_123130057_mpr-2_nifti',
            'OAS2_0036_MR3_2_123130589_mpr-2_nifti','OAS2_0036_MR5_2_123130720_mpr-2_nifti',
            'OAS2_0039_MR2_2_123131109_mpr-2_nifti','OAS2_0069_MR1_1_123135756_mpr-1_nifti',
            'OAS2_0075_MR2_1_123136755_mpr-1_nifti','OAS2_0076_MR3_2_123136956_mpr-2_nifti',
            'OAS2_0081_MR1_1_123137748_mpr-1_nifti','OAS2_0081_MR2_1_123137817_mpr-1_nifti',
            'OAS2_0087_MR1_1_123138148_mpr-1_nifti','OAS2_0088_MR1_2_123138280_mpr-2_nifti',
            'OAS2_0101_MR2_2_123140136_mpr-2_nifti','OAS2_0101_MR3_1_123140201_mpr-1_nifti',
            'OAS2_0106_MR1_3_123140931_mpr-3','OAS2_0106_MR2_4_123141000_mpr-4',
            'OAS2_0112_MR2_2_123141525_mpr-2','OAS2_0120_MR2_2_123142649_mpr-2',
            'OAS2_0126_MR1_2_123143108_mpr-2','OAS2_0126_MR2_4_123143178_mpr-4',
            'OAS2_0129_MR3_3_123143905_mpr-3','OAS2_0134_MR1_2_123144229_mpr-2',
            'OAS2_0139_MR1_1_123144752_mpr-1','OAS2_0140_MR2_1_123144950_mpr-1',
            'OAS2_0140_MR3_3_123145018_mpr-3','OAS2_0142_MR1_1_123145212_mpr-1',
            'OAS2_0142_MR2_4_123145283_mpr-4','OAS2_0145_MR1_2_123145683_mpr-2',
            'OAS2_0145_MR2_3_123145750_mpr-3','OAS2_0147_MR1_4_123145950_mpr-4',
            'OAS2_0147_MR2_3_123146017_mpr-3','OAS2_0147_MR3_2_123146084_mpr-2',
            'OAS2_0147_MR4_2_123146150_mpr-2','OAS2_0149_MR1_2_123146215_mpr-2',
            'OAS2_0149_MR2_4_123146285_mpr-4','OAS2_0150_MR1_1_123146348_mpr-1',
            'OAS2_0150_MR2_4_123146419_mpr-4','OAS2_0152_MR1_2_123146483_mpr-2',
            'OAS2_0152_MR3_2_123146614_mpr-2','OAS2_0154_MR1_2_123146679_mpr-2',
            'OAS2_0154_MR2_2_123146745_mpr-2','OAS2_0158_MR1_2_123147075_mpr-2',
            'OAS2_0162_MR1_4_123147669_mpr-4',
            'OAS2_0169_MR1_4_123148071_mpr-4','OAS2_0169_MR2_3_123148136_mpr-3',
            'OAS2_0182_MR1_2_123149784_mpr-2','OAS2_0183_MR2_1_123149982_mpr-1',
            'OAS2_0183_MR4_3_123150118_mpr-3','OAS2_0184_MR1_3_123150183_mpr-3',
            'OAS2_0184_MR2_3_123150249_mpr-3','OAS2_0186_MR1_4_123150511_mpr-4',
            'OAS2_0186_MR2_1_123150576_mpr-1','OAS2_0186_MR3_3_123150644_mpr-3',
            'ucla_sub-10523_anat_sub-10523_T1w','ucla_sub-10668_anat_sub-10668_T1w',
            'ucla_sub-10697_anat_sub-10697_T1w','ucla_sub-10707_anat_sub-10707_T1w',
            'ucla_sub-10724_anat_sub-10724_T1w','ucla_sub-10746_anat_sub-10746_T1w',
            'ucla_sub-10788_anat_sub-10788_T1w','ucla_sub-10912_anat_sub-10912_T1w',
            'ucla_sub-10948_anat_sub-10948_T1w','ucla_sub-10949_anat_sub-10949_T1w',
            'ucla_sub-10977_anat_sub-10977_T1w','ucla_sub-11019_anat_sub-11019_T1w',
            'ucla_sub-11044_anat_sub-11044_T1w','ucla_sub-11052_anat_sub-11052_T1w',
            'ucla_sub-11061_anat_sub-11061_T1w','ucla_sub-11062_anat_sub-11062_T1w',
            'ucla_sub-11066_anat_sub-11066_T1w','ucla_sub-11067_anat_sub-11067_T1w',
            'ucla_sub-11068_anat_sub-11068_T1w','ucla_sub-11082_anat_sub-11082_T1w',
            'ucla_sub-11106_anat_sub-11106_T1w','ucla_sub-11108_anat_sub-11108_T1w',
            'ucla_sub-50052_anat_sub-50052_T1w','ucla_sub-50056_anat_sub-50056_T1w',
            'ucla_sub-50083_anat_sub-50083_T1w',#'ucla_sub-50085_anat_sub-50085_T1w',
            'ucla_sub-11030_anat_sub-11030_T1w',
            'ucla_sub-60028_anat_sub-60028_T1w'
        ]

        self.subjectIDs = self.subjectIDs_filtered_85
        self.electrode_positions = ['CP4CP5', 'CP4TP7', 'CzCP5', 'CzTP7', 'F3F4']
        self.data_size_1position_1subject = 200

        if self.varification_test:
            self.data_path = data_path
            self.subjectIDs = ['ucla_sub-60028_anat_sub-60028_T1w']
            self.electrode_positions = ['CP4CP5', 'CP4TP7', 'CzCP5', 'CzTP7', 'F3F4']
            self.data_size_1position_1subject = 10

        self.transfer_learning_non_trained_positions = False
        if self.transfer_learning_non_trained_positions:
            self.subjectIDs = ['1_non_trained_positions_20210912/120111',
                               '1_non_trained_positions_20210912/122317',
                               '1_non_trained_positions_20210912/122620',
                               '1_non_trained_positions_20210912/123117',
                               '1_non_trained_positions_20210912/123925',
                               '1_non_trained_positions_20210912/124422',
                               '1_non_trained_positions_20210912/125525',
                               '1_non_trained_positions_20210912/126325',
                               '1_non_trained_positions_20210912/127630',
                               '1_non_trained_positions_20210912/127933',
                               ]
            self.electrode_positions = ['Fp1F4', 'Fp2C3', 'POzCz']
            self.data_size_1position_1subject = 200

        self.test_default_cond = False
        if self.test_default_cond:
            self.subjectIDs = ['120111_20210912/default_cond/']
            self.data_size_1position_1subject = 1

        self.test_MC_samples = False
        if self.test_MC_samples:
            self.subjectIDs = ['OAS2_0145_MR1_2_123145683_mpr-2_20220704_Monte_Carlo_100_rect5050/']
            self.electrode_positions = ['CP4CP5']
            self.data_size_1position_1subject = 100

        self.transfer_learning_ellipse5050 = False
        if self.transfer_learning_ellipse5050:
            self.data_path = '/data/xiaofan/data/tDCS/data_diff_shape_size_20220615/ellipse5050/'
            self.subjectIDs = [
                               '122317_20220615_ellipse5050',
                               '122620_20220615_ellipse5050',
                               '124422_20220615_ellipse5050',
                               '125525_20220615_ellipse5050',
                               '120111_20220615_ellipse5050',
                               '126325_20220615_ellipse5050',
                               '127630_20220615_ellipse5050',
                               '133019_20220615_ellipse5050',
                               '135225_20220615_ellipse5050',
                               '148840_20220615_ellipse5050',
                               ] #10

        self.transfer_learning_rect4040 = False
        if self.transfer_learning_rect4040:
            self.data_path = '/data/xiaofan/data/tDCS/data_diff_shape_size_20220615/rect4040/'
            self.subjectIDs = [
                               '122317_20220615_rect4040',
                               '122620_20220615_rect4040',
                               '124422_20220615_rect4040',
                               '125525_20220615_rect4040',
                                '120111_20220615_rect4040',
                               '126325_20220615_rect4040',
                               '127630_20220615_rect4040',
                               '133019_20220615_rect4040',
                               '135225_20220615_rect4040',
                               '148840_20220615_rect4040',
                               ] #10

        self.num_files = self.data_size_1position_1subject * len(self.subjectIDs) * len(self.electrode_positions)
        info = {
            'subjectIDs': self.subjectIDs,
            'electrode_positions': self.electrode_positions,
            'data_path': self.data_path,
            'data_size_1position_1subject': self.data_size_1position_1subject
        }
        self.save_data_dir = './saved_data/'
        try:
            os.stat(self.save_data_dir)
        except:
            os.mkdir(self.save_data_dir)
        pickle.dump(info, open(os.path.join(self.save_data_dir, 'info_data_load.pkl'), 'wb'))

        np.random.seed(TRAIN_TESTVAL_SEED)  # 0 has to be same for all workers so that train/test/val lists are the same
        self.list_IDs = self.create_file_list()

        self.num_images = self.get_length()

        np.random.seed(self.seed)  # Now seed workers differently so that the sequence is different for each worker
        self.on_epoch_end()  # Generate the sequence

        self.num_batches = self.__len__()

    def get_length(self):
        """
        Get the length of the list of file IDs associated with this data loader
        """
        return len(self.list_IDs)

    def get_file_list(self):
        """
        Get the list of file IDs associated with this data loader
        """
        return self.list_IDs

    def print_info(self):
        """
        Print the dataset information
        """
        print("*" * 30)
        print("=" * 30)
        print("Number of {} images = {}".format(self.setType, self.num_images))
        print("=" * 30)
        print("*" * 30)

    def create_file_list(self):
        """
        Get list of the files from the BraTS raw data
        Split into training and testing sets.
        """
        numFiles = self.num_files
        idxList = np.arange(numFiles)  # List of file indices

        data_size_nposition_1subject = self.data_size_1position_1subject * len(self.electrode_positions)
        subjectID_shuffled = np.random.permutation(np.arange(0, len(self.subjectIDs)))  # [3,1,2,0,4]
        print('\nsubjectID shuffled:', subjectID_shuffled, '\n')

        for i, subjectid in enumerate(subjectID_shuffled):
            idxList[i * data_size_nposition_1subject:(i + 1) * data_size_nposition_1subject] \
                = subjectid * data_size_nposition_1subject + np.arange(0, data_size_nposition_1subject)
        if self.varification_test:
            self.trainIdx = []
            self.validateIdx = []
            self.testIdx = idxList 
        elif self.test_default_cond or self.test_MC_samples:
            self.trainIdx = []
            self.validateIdx = []
            self.testIdx = idxList
        elif self.transfer_learning_non_trained_positions or \
            self.transfer_learning_ellipse5050 or self.transfer_learning_rect4040:
            self.trainIdx = idxList[0:(-3 * data_size_nposition_1subject)]  # List of training indices
            self.validateIdx = idxList[(-3 * data_size_nposition_1subject):(
                        -2 * data_size_nposition_1subject)]  # List of validation indices
            self.testIdx = idxList[
                           (-2 * data_size_nposition_1subject):]  # List of testing indices (last testIdx elements)
        elif len(subjectID_shuffled) == 85:
            self.trainIdx = idxList[0:(-26 * data_size_nposition_1subject)]  # List of training indices
            self.validateIdx = idxList[(-26 * data_size_nposition_1subject):(
                        -13 * data_size_nposition_1subject)]  # List of validation indices
            self.testIdx = idxList[
                           (-13 * data_size_nposition_1subject):]  # List of testing indices (last testIdx elements)
        else:
            num_train_MRI =  np.floor((len(subjectID_shuffled))*0.7)
            num_val_MRI = np.floor((len(subjectID_shuffled))*0.15)
            self.trainIdx = idxList[0:(num_train_MRI * data_size_nposition_1subject)]  # List of training indices
            self.validateIdx = idxList[(num_train_MRI * data_size_nposition_1subject):(
                        (num_train_MRI+num_val_MRI) * data_size_nposition_1subject)]  # List of validation indices
            self.testIdx = idxList[
                           ((num_train_MRI+num_val_MRI)* data_size_nposition_1subject):]  # List of testing indices (last testIdx elements)


        IDX = [self.trainIdx, self.validateIdx, self.testIdx]  # the min of IDX is 0
        pickle.dump(IDX, open(os.path.join(self.save_data_dir, 'IDX.pkl'), 'wb'))

        if self.setType == "train":
            return self.trainIdx
        elif self.setType == "validate":
            return self.validateIdx
        elif self.setType == "test":
            return self.testIdx
        else:
            print("Error. You forgot to specify train, test, or validate. Instead received {}".format(self.setType))
            return []

    def __len__(self):
        """
        The number of batches per epoch
        """
        return self.num_images // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indicies of the batch
        indexes = np.sort(
            self.indexes[(index * self.batch_size):((index + 1) * self.batch_size)])

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return (X, y)

    def get_batch(self, index):
        """
        Public method to get one batch of data
        """
        return self.__getitem__(index)

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        If shuffle is true, then it will shuffle the training set
        after every epoch.
        """
        self.indexes = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_batch_fileIDs(self):
        """
        Get the file IDs of the last batch that was loaded.
        """
        return self.fileIDs

    def get_one_sample(self, fileIdx):
        num_electrode_position = len(self.electrode_positions)
        num_samples_nposition_1subject = num_electrode_position * self.data_size_1position_1subject
        subjectID = self.subjectIDs[fileIdx // (num_samples_nposition_1subject)]

        subfolder = self.electrode_positions[
            (fileIdx % num_samples_nposition_1subject) // self.data_size_1position_1subject]
        subsubfolder_int = (fileIdx % (num_samples_nposition_1subject) + 1) % self.data_size_1position_1subject
        if not subsubfolder_int:
            subsubfolder_int = self.data_size_1position_1subject

        foldername = self.data_path + subjectID + '/' + subfolder + '/' + str(subsubfolder_int) + '/'

        if not os.path.exists(foldername):
            print(foldername + 'does not exist!')

        filename = 'field_cond.mat'
        try:
            mat = loadmat(foldername + filename)
        except:
            print('DeeptDCS Error: load mat failed.\n', foldername, filename)
        img = np.asarray(mat['voxelConductivity'])

        msk = np.asarray(mat['headVoxelJfield'])

        # add dimension to img
        if len(np.shape(img)) == 3:
            img = np.expand_dims(img, axis=3)

        input_weighting_scheme = True  # actually super necessary, as target in the data purdue set has electrodes
        if input_weighting_scheme:
            weights = img != 0
            weights[img == 2] = 0
            weights[img == 3] = 0
            weights = np.repeat(weights, np.shape(msk)[-1], axis=-1)
            msk = np.multiply(weights, msk)

        voxRegion = np.asarray(mat['voxelRegions'])  # unique: 0,1,2,3,4,5,6,501,502
        voxRegion = np.expand_dims(voxRegion, -1)

        msk_voxRegion = np.concatenate((msk, voxRegion), -1)
        return img, msk_voxRegion

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples

        This just reads the list of filename to load.
        Change this to suit your dataset.
        """

        # Make empty arrays for the images and mask batches
        imgs = np.zeros((self.batch_size, *self.dim, self.n_in_channels))
        msk_voxRegions = np.zeros(
            (self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.n_out_channels + self.n_in_channels))

        for idx, fileIdx in enumerate(list_IDs_temp):
            img, msk_voxRegion, = self.get_one_sample(fileIdx)

            imgs[idx,] = img
            msk_voxRegions[idx,] = msk_voxRegion

        return imgs, msk_voxRegions

    def get_test_inputs_targets(self):
        imgs = np.zeros((len(self.testIdx), *self.dim, self.n_in_channels))
        targets = np.zeros((len(self.testIdx), self.dim[0], self.dim[1], self.dim[2], self.n_out_channels))
        for idx, fileIdx in enumerate(self.testIdx):
            img, msk_voxRegion = self.get_one_sample(fileIdx)

            target = msk_voxRegion[:, :, :, 0:3]  # with electrodes
            voxRegion = msk_voxRegion[:, :, :, -1]

            # filter out electrodes from targets
            head_msk = (voxRegion != 0) & (voxRegion < 500)  # weight [144,144,144]
            head_msk = np.expand_dims(head_msk, axis=-1)
            head_m = np.repeat(head_msk, repeats=3, axis=-1)
            target = target * head_m  # without electrodes

            imgs[idx,] = img
            targets[idx,] = target
        return imgs, targets

    def get_test_targets_max_abs(self):
        max_abs = 0
        for idx, fileIdx in enumerate(self.testIdx):
            if idx % 50 == 0:
                print(idx + 1, '/', len(self.testIdx), '...\n')
            _, msk_voxRegion = self.get_one_sample(fileIdx)

            target = msk_voxRegion[:, :, :, 0:3]
            max_abs = np.maximum(max_abs, np.max(np.abs(target)))
        return max_abs

    def save_test_set(self, test_inp_set, test_targ_set):
        print('Saving test_inputs.')
        with gzip.open(os.path.join(self.save_data_dir, 'test_inputs.pkl'), 'wb') as f:
            pickled = pickle.dumps(test_inp_set, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        print('Saving test_targets.')
        with gzip.open(os.path.join(self.save_data_dir, 'test_targets.pkl'), 'wb') as f:
            pickled = pickle.dumps(test_targ_set, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)


