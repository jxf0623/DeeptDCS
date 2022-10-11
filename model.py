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

import tensorflow.compat.v1 as tf

from argparser import args
from tensorflow.keras.utils import multi_gpu_model, plot_model
import pickle
import os
import numpy as np

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

tf.compat.v1.enable_eager_execution()


class unet(object):

    def __init__(self, use_upsampling=False, learning_rate=0.001,
                 n_cl_in=1, n_cl_out=1, feature_maps = 16,
                 dropout=0.2, print_summary=False,
                 channels_last = True, batch_size = 2, num_gpu = 0,
                 height=144, width=144, depth=144,
                 ):
        self.bz = batch_size
        self.num_gpu = num_gpu
        self.channels_last = channels_last

        if channels_last:
            self.concat_axis = -1
            self.data_format = "channels_last"

        else:
            self.concat_axis = 1
            self.data_format = "channels_first"

        K.backend.set_image_data_format(self.data_format)

        self.fms = feature_maps # 16 or 32 feature maps in the first convolutional layer

        self.use_upsampling = use_upsampling
        self.print_summary = print_summary
        self.n_cl_in = n_cl_in
        self.n_cl_out = n_cl_out
        self.height = height
        self.width = width
        self.depth = depth

        self.loss = self.mean_squared_error

        self.metrics = [self.MRDM,self.RE,self.MAE_J,self.ml2e_head]

        self.learning_rate = learning_rate
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate)

        # self.model = self.unet_3d(self.num_gpu)
        # self.model = self.MSResUnet_3d(self.num_gpu)
        # self.model = self.ResUnet_3d(self.num_gpu)
        self.model = self.attn_unet_3d(self.num_gpu)
        # self.model = self.attn_ResUnet_3d(self.num_gpu)

        self.save_all_test_error = False

    def MRDM (self,target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]

        pd = prediction.numpy()  # (bz,144,144,144,3)
        tg = target.numpy()  # (bz,144,144,144,3)

        if tf.count_nonzero(target) != tf.count_nonzero(prediction):
            voxRegion = target_voxRegion[:, :, :, :, -1]
            head_msk = np.where(voxRegion > 0, 1, 0) & np.where(voxRegion < 500, 1, 0)  # weight [bz,144,144,144]
            head_msk_3d = np.repeat(np.expand_dims(head_msk, axis=-1), 3, axis=-1)
            tg = tg * head_msk_3d  # without electrodes

        tg_normJ = np.linalg.norm(tg, axis=-1)  # (bz,144,144,144)
        tg_normJ_expand = np.expand_dims(tg_normJ,axis=-1)  #(bz,144,144,144,1)
        tg_normJ_repeat3 = np.repeat(tg_normJ_expand,3,axis=-1)  #(bz,144,144,144,3)
        normalized_tg_w_nan = tg/tg_normJ_repeat3  #(bz,144,144,144,3)
        normalized_tg_wo_nan = np.nan_to_num(normalized_tg_w_nan,nan=0) #nan to zero #(bz,144,144,144,3)

        pd_normJ = np.linalg.norm(pd,axis=-1)  #(bz,144,144,144)
        pd_normJ_expand = np.expand_dims(pd_normJ,axis=-1)  #(bz,144,144,144,1)
        pd_normJ_repeat3 = np.repeat(pd_normJ_expand,3,axis=-1)  #(bz,144,144,144,3)
        normalized_pd_w_nan = pd/pd_normJ_repeat3  #(bz,144,144,144,3)
        normalized_pd_wo_nan = np.nan_to_num(normalized_pd_w_nan,nan=0) #nan to zero #(bz,144,144,144,3)

        diff_normalized_tg_pd_wo_nan = normalized_tg_wo_nan-normalized_pd_wo_nan #(bz,144,144,144,3)
        voxel_level_RDM = np.linalg.norm(diff_normalized_tg_pd_wo_nan,axis = -1) #(bz,144,144,144)

        batch_size = np.shape(tg)[0]
        error_batch_sample = np.zeros(batch_size)
        for i in range(0, batch_size):
            voxel_level_RDM_single_sample = voxel_level_RDM[i,:,:,:]
            error_single_sample = np.sum(voxel_level_RDM_single_sample)/np.count_nonzero(voxel_level_RDM_single_sample)
            error_batch_sample[i] = error_single_sample

        average_error = np.mean(error_batch_sample)

        if self.save_all_test_error:
            self.save_data_dir = './saved_data/'
            try:
                all_test_error = pickle.load(
                    open(os.path.join(self.save_data_dir, 'all_test_error_MRDM.pkl'), 'rb'))
            except:
                all_test_error = []
            all_test_error.append(error_batch_sample)
            pickle.dump(all_test_error, open(os.path.join(self.save_data_dir, 'all_test_error_MRDM.pkl'), 'wb'))
        return average_error

    def RE(self,target, prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]

        pd = prediction.numpy() #(bz,144,144,144,3)
        tg = target.numpy() #(bz,144,144,144,3)

        if tf.count_nonzero(target)!=tf.count_nonzero(prediction):
            voxRegion = target_voxRegion[:, :, :, :, -1]
            head_msk = np.where(voxRegion>0,1,0) & np.where(voxRegion<500,1,0) # weight [bz,144,144,144]
            head_msk_3d = np.repeat(np.expand_dims(head_msk,axis=-1),3,axis=-1)
            tg = tg*head_msk_3d #without electrodes

        tg_pd_diff = tg-pd #(bz,144,144,144,3)
        tg_normJ = np.linalg.norm(tg, axis=-1) #(bz,144,144,144)
        tg_pd_diff_normJ = np.linalg.norm(tg_pd_diff, axis=-1) #(bz,144,144,144)

        batch_size = np.shape(tg_normJ)[0]
        error_batch_sample=np.zeros(batch_size)
        for i in range(0,batch_size):
            tg_pd_diff_normJ_sum_single_sample = np.sum(tg_pd_diff_normJ[i,:,:,:])
            tg_normJ_sum_single_sample = np.sum(tg_normJ[i,:,:,:])
            error_single_sample = tg_pd_diff_normJ_sum_single_sample/tg_normJ_sum_single_sample
            error_batch_sample[i] = error_single_sample
        average_error = np.mean(error_batch_sample)

        if self.save_all_test_error:
            self.save_data_dir = './saved_data/'
            try:
                all_test_error = pickle.load(open(os.path.join(self.save_data_dir, 'RE_144_144_144.pkl'),'rb'))
            except:
                all_test_error=[]
            all_test_error.append(error_batch_sample)
            pickle.dump(all_test_error, open(os.path.join(self.save_data_dir, 'RE_144_144_144.pkl'), 'wb'))
        return average_error

    def MAE_J(self,target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        if tf.count_nonzero(target)!=tf.count_nonzero(prediction):
            voxRegion = target_voxRegion[:,:,:,:,-1]
            head_msk = (voxRegion != 0) & (voxRegion < 500)  # weight [bz,144,144,144]
            head_msk = tf.cast(head_msk,dtype=tf.float32)
            head_msk = tf.expand_dims(head_msk,axis=-1)
            head_m = tf.repeat(head_msk,repeats=3,axis=-1)
            target = target * head_m  # without electrodes
        MAE = tf.reduce_mean(tf.abs(target-prediction),axis=(1,2,3,4))
        return MAE

    def ml2e_head(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]

        pd = prediction.numpy()
        tg = target.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1) #(bz,144,144,144) #!without electrodes

        if tf.count_nonzero(target)!=tf.count_nonzero(prediction):
            voxRegion = target_voxRegion[:, :, :, :, -1]
            head_msk = np.where(voxRegion>0,1,0) & np.where(voxRegion<500,1,0) # weight [bz,144,144,144]
            weight = head_msk
        else:
            weight = 1
        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)

        if self.save_all_test_error:
            self.save_data_dir = './saved_data/'
            try:
                all_test_error = pickle.load(open(os.path.join(self.save_data_dir, 'all_test_error_ml2e_head.pkl'),'rb'))
            except:
                all_test_error=[]
            all_test_error.append(error_batch_sample)
            pickle.dump(all_test_error, open(os.path.join(self.save_data_dir, 'all_test_error_ml2e_head.pkl'), 'wb'))

        return average_error

    def ml2e_tissue1(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 1
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)

        return average_error

    def ml2e_tissue2(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 2
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def ml2e_tissue3(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 3
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def ml2e_tissue4(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 4
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def ml2e_tissue5(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 5
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def ml2e_tissue6(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        tissue_idx = 6
        weight = np.where(voxRegion==tissue_idx,1,0)#weight [bz,144,144,144]

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def ml2e_tissue12_wm_gm(self, target,prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        voxRegion = target_voxRegion[:,:,:,:,-1]

        tg = target.numpy()
        pd = prediction.numpy()

        tg_normJ = np.linalg.norm(tg,axis=-1)
        pd_normJ = np.linalg.norm(pd,axis=-1)

        weight1 = np.where(voxRegion==1,1,0) #weight1[bz,144,144,144]
        weight2 = np.where(voxRegion==2,1,0) #weight2[bz,144,144,144]
        weight = weight1+weight2

        error_batch_sample = np.sqrt(np.sum(np.square(tg_normJ-pd_normJ)*weight,axis=(1,2,3))/np.sum(np.square(tg_normJ)*weight,axis=(1,2,3))) #(bz)
        average_error = np.average(error_batch_sample)
        return average_error

    def mean_squared_error(self, target, prediction):
        target_voxRegion = target
        target = target_voxRegion[:, :, :, :, 0:3]
        return tf.losses.mean_squared_error(target, prediction)

    def unet_3d(self, num_gpu):
        """
        3D U-Net
        """

        def ConvolutionBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 3
            if num_blk ==2:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name)(x)

            if num_blk ==3:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name+"_relu1")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x)
                x = K.layers.BatchNormalization(name=name+"_bn2")(x)
                x = K.layers.Activation("relu", name=name)(x)
            return x


        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ConvolutionBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ConvolutionBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ConvolutionBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ConvolutionBlock(poolC, "encodeD", self.fms*8, params)

        num_encoder = 4
        if num_encoder == 4:
            poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

            encodeE = ConvolutionBlock(poolD, "encodeE", self.fms*16, params)
            # END - Encoding path

            # BEGIN - Decoding path
            if self.use_upsampling:
                up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                           interpolation="bilinear")(encodeE)
            else:
                up = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                              **params_trans)(encodeE)
            concatD = K.layers.concatenate(
                [up, encodeD], axis=self.concat_axis, name="concatD")

            decodeC = ConvolutionBlock(concatD, "decodeC", self.fms*8, params)
        elif num_encoder == 3:
            decodeC = encodeD
        else:
            print('Error: The number of encoders are invalid!')

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ConvolutionBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ConvolutionBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = ConvolutionBlock(concatA, "convOut", self.fms, params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(convOut)
        # prediction = tf.Print(prediction,[prediction], '\nprediction is:')
        input_weighting_scheme = True
        if input_weighting_scheme:
            weights = (inputs !=0) & (inputs < 2)
            weights = tf.repeat(weights,tf.shape(prediction)[-1],axis=-1)
            weights = tf.cast(weights,tf.float32)
            prediction = K.layers.Multiply()([weights, prediction])

        if num_gpu <= 1:
            print('[INFO] training with 1 GPU/ CPU ...')
            model = K.models.Model(inputs=[inputs], outputs=[prediction])

        else:
            print('[INFO] training with {} GPUs ...'.format(num_gpu))
            with tf.device('/cpu:0'):
                model = K.models.Model(inputs=[inputs], outputs=[prediction])
            model = multi_gpu_model(model, num_gpu)

        if self.print_summary:
            model.summary()

        plot_model(model, to_file = '3dunet.png')

        return model

    def attn_unet_3d(self, num_gpu):
        """
        attn 3D U-Net as in Attn U-net, different from IJCNN TMS paper

        """

        def attn_blk(x_upper,x_lower,fms):
            x_upper_1 = K.layers.Conv3D (filters=fms, kernel_size=(2,2,2),strides=(2,2,2),use_bias=False)(x_upper)
            x_upper_2 = K.layers.BatchNormalization()(x_upper_1)

            x_lower_1 = K.layers.Conv3D (filters=fms, kernel_size=(1,1,1),strides=(1,1,1),use_bias=True)(x_lower)
            x_lower_2 = K.layers.BatchNormalization()(x_lower_1)

            x3 = K.layers.Add()([x_upper_2,x_lower_2])
            x3 = K.layers.Activation('relu')(x3)

            x4 = K.layers.Conv3D(filters=1,kernel_size=(1,1,1),strides=(1,1,1))(x3)
            x4 = K.layers.BatchNormalization()(x4)
            x4 = K.layers.Activation("sigmoid")(x4)

            x5 = K.layers.UpSampling3D(size=(2,2,2))(x4)
            x5 = tf.multiply(x5,x_upper)

            return x5


        def ConvolutionBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 3
            if num_blk ==2:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name)(x)

            if num_blk ==3:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name+"_relu1")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x)
                x = K.layers.BatchNormalization(name=name+"_bn2")(x)
                x = K.layers.Activation("relu", name=name)(x)
            return x


        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ConvolutionBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ConvolutionBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ConvolutionBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ConvolutionBlock(poolC, "encodeD", self.fms*8, params)

        num_encoder = 4
        if num_encoder == 4:
            poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

            encodeE = ConvolutionBlock(poolD, "encodeE", self.fms*16, params)
            # END - Encoding path

            # BEGIN - Decoding path
            if self.use_upsampling:
                up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                           interpolation="bilinear")(encodeE)
            else:
                up = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                              **params_trans)(encodeE)

            encodeD = attn_blk(encodeD,encodeE,self.fms*8)

            concatD = K.layers.concatenate(
                [up, encodeD], axis=self.concat_axis, name="concatD")

            decodeC = ConvolutionBlock(concatD, "decodeC", self.fms*8, params)
        elif num_encoder == 3:
            decodeC = encodeD
        else:
            print('Error: The number of encoders are invalid!')

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)

        encodeC = attn_blk(encodeC,decodeC,self.fms*4)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ConvolutionBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)

        encodeB = attn_blk(encodeB,decodeB,self.fms*2)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ConvolutionBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)

        encodeA = attn_blk(encodeA,decodeA,self.fms)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = ConvolutionBlock(concatA, "convOut", self.fms, params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(convOut)
        # prediction = tf.Print(prediction,[prediction], '\nprediction is:')
        input_weighting_scheme = True
        if input_weighting_scheme:
            weights = (inputs !=0) & (inputs < 2)
            weights = tf.repeat(weights,tf.shape(prediction)[-1],axis=-1)
            weights = tf.cast(weights,tf.float32)
            prediction = K.layers.Multiply()([weights, prediction])

        if num_gpu <= 1:
            print('[INFO] training with 1 GPU/ CPU ...')
            model = K.models.Model(inputs=[inputs], outputs=[prediction])

        else:
            print('[INFO] training with {} GPUs ...'.format(num_gpu))
            with tf.device('/cpu:0'):
                model = K.models.Model(inputs=[inputs], outputs=[prediction])
            model = multi_gpu_model(model, num_gpu)

        if self.print_summary:
            model.summary()

        plot_model(model, to_file = 'attn_unet_3d.png')

        return model

    def MSResUnet_3d(self, num_gpu):
        """
        3D MSResUnet
        """


        def ConvolutionBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 2
            if num_blk ==2:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name)(x)

            if num_blk ==3:
                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x = K.layers.BatchNormalization(name=name+"_bn0")(x)
                x = K.layers.Activation("relu", name=name+"_relu0")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
                x = K.layers.BatchNormalization(name=name+"_bn1")(x)
                x = K.layers.Activation("relu", name=name+"_relu1")(x)

                x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x)
                x = K.layers.BatchNormalization(name=name+"_bn2")(x)
                x = K.layers.Activation("relu", name=name)(x)
            return x

        def ResidualBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 3
            ResNetV1 = 0
            ResNetV2 = 1
            if num_blk ==2:
                #TODO not working due to dim mismatch
                if ResNetV1:
                    x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                    x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                    x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                    x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                    x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)

                    x = K.layers.Add()([x,x2_2])
                    x = K.layers.Activation("relu", name=name)(x)
                elif ResNetV2:
                    x1_1 = K.layers.BatchNormalization(name=name + "_bn0")(x)
                    x1_2 = K.layers.Activation("relu", name=name + "_relu0")(x1_1)
                    x1_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv0")(x1_2)

                    x2_1 = K.layers.BatchNormalization(name=name + "_bn1")(x1_3)
                    x2_2 = K.layers.Activation("relu", name=name)(x2_1)
                    x2_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv1")(x2_2)

                    x = K.layers.Add()([x,x2_3])
                return x
            elif num_blk ==3: # as in Qiqi's paper
                x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)
                x2_3 = K.layers.Activation("relu", name=name+"_relu1")(x2_2)

                x3_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x2_3)
                x3_2 = K.layers.BatchNormalization(name=name+"_bn2")(x3_1)
                x3_3 = K.layers.Add()([x1_2,x3_2])
                x3_4 = K.layers.Activation("relu", name=name)(x3_3)
                return x3_4

        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ResidualBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ResidualBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ResidualBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ResidualBlock(poolC, "encodeD", self.fms*8, params)

        num_encoder = 4
        if num_encoder == 4:
            poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

            encodeE = ResidualBlock(poolD, "encodeE", self.fms*16, params)
            # END - Encoding path

            # BEGIN - Decoding path
            if self.use_upsampling:
                upE = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                           interpolation="bilinear")(encodeE)
            else:
                upE = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                              **params_trans)(encodeE)
            concatD = K.layers.concatenate(
                [upE, encodeD], axis=self.concat_axis, name="concatD")

            decodeC = ConvolutionBlock(concatD, "decodeC", self.fms*8, params)
        elif num_encoder == 3:
            decodeC = encodeD
        else:
            print('Error: The number of encoders are invalid!')

        if self.use_upsampling:
            upC = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            upC = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [upC, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ConvolutionBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            upB = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            upB = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [upB, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ConvolutionBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            upA = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            upA = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [upA, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path
        convOut1 = K.layers.Conv3D(filters=self.fms, **params, name='convOut1' + "_conv0")(concatA)
        convOut1 = K.layers.BatchNormalization(name='convOut1' + "_bn0")(convOut1)
        convOut1 = K.layers.Activation("relu", name='convOut1' + "_relu0")(convOut1)
        convOut1 = K.layers.Conv3D(filters=self.n_cl_out, **params, name='convOut1' + "_conv1")(convOut1)
        convOut1 = K.layers.BatchNormalization(name='convOut1' + "_bn1")(convOut1)
        convOut1 = K.layers.Activation("relu", name='convOut1'+"_relu1")(convOut1)

        decodeA_1X1X1 = K.layers.Conv3D(name="decodeA_1X1X1_conv",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(decodeA)
        decodeA_1X1X1 = K.layers.BatchNormalization(name="decodeA_1X1X1"+'_BN')(decodeA_1X1X1)

        decodeB_up = K.layers.Conv3D(name="decodeB_up_conv",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(decodeB)
        decodeB_up = K.layers.BatchNormalization(name="decodeB_up"+'_BN0')(decodeB_up)
        decodeB_up = K.layers.Conv3DTranspose(name="decodeB_up_deconv", filters=self.n_cl_out,
                                       **params_trans)(decodeB_up)
        decodeB_up = K.layers.BatchNormalization(name="decodeB_up"+'_BN1')(decodeB_up)

        addOut2 = decodeA_1X1X1+decodeB_up
        convOut2_up = K.layers.Conv3DTranspose(name="convOut2_up", filters=self.n_cl_out,
                                       **params_trans)(addOut2)
        addOut = convOut1+convOut2_up

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(addOut)

        input_weighting_scheme = True
        if input_weighting_scheme:
            weights = (inputs !=0) & (inputs < 2)
            weights = tf.repeat(weights,tf.shape(prediction)[-1],axis=-1)
            weights = tf.cast(weights,tf.float32)
            prediction = K.layers.Multiply()([weights, prediction])

        if num_gpu <= 1:
            print('[INFO] training with 1 GPU/ CPU ...')
            model = K.models.Model(inputs=[inputs], outputs=[prediction])

        else:
            print('[INFO] training with {} GPUs ...'.format(num_gpu))
            with tf.device('/cpu:0'):
                model = K.models.Model(inputs=[inputs], outputs=[prediction])
            model = multi_gpu_model(model, num_gpu)

        if self.print_summary:
            model.summary()

        plot_model(model, to_file = 'MSResUnet_3d.png')

        return model

    def ResUnet_3d(self, num_gpu):
        """
        3D ResUnet

        replace convolutional block in unet_3d with ResidualBlock
        """

        def ResidualBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 3
            ResNetV1 = 0
            ResNetV2 = 1
            if num_blk ==2:
                #TODO not working due to dim mismatch
                if ResNetV1:
                    x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                    x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                    x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                    x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                    x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)

                    x = K.layers.Add()([x,x2_2])
                    x = K.layers.Activation("relu", name=name)(x)
                elif ResNetV2:
                    x1_1 = K.layers.BatchNormalization(name=name + "_bn0")(x)
                    x1_2 = K.layers.Activation("relu", name=name + "_relu0")(x1_1)
                    x1_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv0")(x1_2)

                    x2_1 = K.layers.BatchNormalization(name=name + "_bn1")(x1_3)
                    x2_2 = K.layers.Activation("relu", name=name)(x2_1)
                    x2_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv1")(x2_2)

                    x = K.layers.Add()([x,x2_3])
                return x
            elif num_blk ==3: # as in Qiqi's paper
                x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)
                x2_3 = K.layers.Activation("relu", name=name+"_relu1")(x2_2)

                x3_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x2_3)
                x3_2 = K.layers.BatchNormalization(name=name+"_bn2")(x3_1)
                x3_3 = K.layers.Add()([x1_2,x3_2])
                x3_4 = K.layers.Activation("relu", name=name)(x3_3)
                return x3_4


        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ResidualBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ResidualBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ResidualBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ResidualBlock(poolC, "encodeD", self.fms*8, params)

        num_encoder = 4
        if num_encoder == 4:
            poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

            encodeE = ResidualBlock(poolD, "encodeE", self.fms*16, params)
            # END - Encoding path

            # BEGIN - Decoding path
            if self.use_upsampling:
                up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                           interpolation="bilinear")(encodeE)
            else:
                up = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                              **params_trans)(encodeE)
            concatD = K.layers.concatenate(
                [up, encodeD], axis=self.concat_axis, name="concatD")

            decodeC = ResidualBlock(concatD, "decodeC", self.fms*8, params)
        elif num_encoder == 3:
            decodeC = encodeD
        else:
            print('Error: The number of encoders are invalid!')

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ResidualBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ResidualBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = ResidualBlock(concatA, "convOut", self.fms, params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(convOut)

        input_weighting_scheme = True
        if input_weighting_scheme:
            weights = (inputs !=0) & (inputs < 2)
            weights = tf.repeat(weights,tf.shape(prediction)[-1],axis=-1)
            weights = tf.cast(weights,tf.float32)
            prediction = K.layers.Multiply()([weights, prediction])

        if num_gpu <= 1:
            print('[INFO] training with 1 GPU/ CPU ...')
            model = K.models.Model(inputs=[inputs], outputs=[prediction])

        else:
            print('[INFO] training with {} GPUs ...'.format(num_gpu))
            with tf.device('/cpu:0'):
                model = K.models.Model(inputs=[inputs], outputs=[prediction])
            model = multi_gpu_model(model, num_gpu)

        if self.print_summary:
            model.summary()

        plot_model(model, to_file = 'ResUnet_3d.png')

        return model

    def attn_ResUnet_3d(self, num_gpu):
        """
        attn 3D U-Net as in Attn U-net, different from IJCNN TMS paper
        replace convolutional block in attn_unet_3d with ResidualBlock

        """

        def attn_blk(x_upper,x_lower,fms):
            x_upper_1 = K.layers.Conv3D (filters=fms, kernel_size=(2,2,2),strides=(2,2,2),use_bias=False)(x_upper)
            x_upper_2 = K.layers.BatchNormalization()(x_upper_1)

            x_lower_1 = K.layers.Conv3D (filters=fms, kernel_size=(1,1,1),strides=(1,1,1),use_bias=True)(x_lower)
            x_lower_2 = K.layers.BatchNormalization()(x_lower_1)

            x3 = K.layers.Add()([x_upper_2,x_lower_2])
            x3 = K.layers.Activation('relu')(x3)

            x4 = K.layers.Conv3D(filters=1,kernel_size=(1,1,1),strides=(1,1,1))(x3)
            x4 = K.layers.BatchNormalization()(x4)
            x4 = K.layers.Activation("sigmoid")(x4)

            x5 = K.layers.UpSampling3D(size=(2,2,2))(x4)
            x5 = tf.multiply(x5,x_upper)

            return x5

        def ResidualBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            num_blk = 3
            ResNetV1 = 0
            ResNetV2 = 1
            if num_blk ==2:
                #TODO not working due to dim mismatch
                if ResNetV1:
                    x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                    x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                    x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                    x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                    x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)

                    x = K.layers.Add()([x,x2_2])
                    x = K.layers.Activation("relu", name=name)(x)
                elif ResNetV2:
                    x1_1 = K.layers.BatchNormalization(name=name + "_bn0")(x)
                    x1_2 = K.layers.Activation("relu", name=name + "_relu0")(x1_1)
                    x1_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv0")(x1_2)

                    x2_1 = K.layers.BatchNormalization(name=name + "_bn1")(x1_3)
                    x2_2 = K.layers.Activation("relu", name=name)(x2_1)
                    x2_3 = K.layers.Conv3D(filters=fms, **params, name=name + "_conv1")(x2_2)

                    x = K.layers.Add()([x,x2_3])
                return x
            elif num_blk ==3: # as in Qiqi's paper
                x1_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
                x1_2 = K.layers.BatchNormalization(name=name+"_bn0")(x1_1)
                x1_3 = K.layers.Activation("relu", name=name+"_relu0")(x1_2)

                x2_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x1_3)
                x2_2 = K.layers.BatchNormalization(name=name+"_bn1")(x2_1)
                x2_3 = K.layers.Activation("relu", name=name+"_relu1")(x2_2)

                x3_1 = K.layers.Conv3D(filters=fms, **params, name=name+"_conv2")(x2_3)
                x3_2 = K.layers.BatchNormalization(name=name+"_bn2")(x3_1)
                x3_3 = K.layers.Add()([x1_2,x3_2])
                x3_4 = K.layers.Activation("relu", name=name)(x3_3)
                return x3_4

        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ResidualBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ResidualBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ResidualBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ResidualBlock(poolC, "encodeD", self.fms*8, params)

        num_encoder = 4
        if num_encoder == 4:
            poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

            encodeE = ResidualBlock(poolD, "encodeE", self.fms*16, params)
            # END - Encoding path

            # BEGIN - Decoding path
            if self.use_upsampling:
                up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                           interpolation="bilinear")(encodeE)
            else:
                up = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                              **params_trans)(encodeE)

            encodeD = attn_blk(encodeD,encodeE,self.fms*8)

            concatD = K.layers.concatenate(
                [up, encodeD], axis=self.concat_axis, name="concatD")

            decodeC = ResidualBlock(concatD, "decodeC", self.fms*8, params)
        elif num_encoder == 3:
            decodeC = encodeD
        else:
            print('Error: The number of encoders are invalid!')

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)

        encodeC = attn_blk(encodeC,decodeC,self.fms*4)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ResidualBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)

        encodeB = attn_blk(encodeB,decodeB,self.fms*2)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ResidualBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)

        encodeA = attn_blk(encodeA,decodeA,self.fms)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = ResidualBlock(concatA, "convOut", self.fms, params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation=None)(convOut)

        input_weighting_scheme = True
        if input_weighting_scheme:
            weights = (inputs !=0) & (inputs < 2)
            weights = tf.repeat(weights,tf.shape(prediction)[-1],axis=-1)
            weights = tf.cast(weights,tf.float32)
            prediction = K.layers.Multiply()([weights, prediction])

        if num_gpu <= 1:
            print('[INFO] training with 1 GPU/ CPU ...')
            model = K.models.Model(inputs=[inputs], outputs=[prediction])

        else:
            print('[INFO] training with {} GPUs ...'.format(num_gpu))
            with tf.device('/cpu:0'):
                model = K.models.Model(inputs=[inputs], outputs=[prediction])
            model = multi_gpu_model(model, num_gpu)

        if self.print_summary:
            model.summary()

        plot_model(model, to_file = 'attn_ResUnet_3d.png')

        return model
