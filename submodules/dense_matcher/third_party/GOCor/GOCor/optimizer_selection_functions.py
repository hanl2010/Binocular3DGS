# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import global_gocor, local_gocor


def define_optimizer_global_corr(global_gocor_arguments):
    """
    Defines the GlobalGOCor optimizer module based on the input parameter dictionary.
    default = {'optim_iter':3, 'num_features': 512, 'init_step_length': 1.0,
               'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5,
               'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
               'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
               'apply_query_loss': False, 'reg_kernel_size': 3, 'reg_inter_dim': 1, 'reg_output_dim': 1.0}
    """

    gocor_arguments_keys = global_gocor_arguments.keys()
    num_features = global_gocor_arguments['num_features'] if 'num_features' in gocor_arguments_keys else 512
    num_iter = global_gocor_arguments['optim_iter'] if 'optim_iter' in gocor_arguments_keys else 3
    init_step_length = global_gocor_arguments['init_step_length'] if 'init_step_length' in gocor_arguments_keys else 1.0
    init_filter_reg = global_gocor_arguments['init_filter_reg'] if 'init_filter_reg' in gocor_arguments_keys else 1e-2
    min_filter_reg = global_gocor_arguments['min_filter_reg'] if 'min_filter_reg' in gocor_arguments_keys else 1e-5
    steplength_reg = global_gocor_arguments['steplength_reg'] if 'steplength_reg' in gocor_arguments_keys else 0.0

    # reference loss parameters
    num_dist_bins = global_gocor_arguments['num_dist_bins'] if 'num_dist_bins' in gocor_arguments_keys else 10
    bin_displacement = global_gocor_arguments['bin_displacement'] if 'bin_displacement' in gocor_arguments_keys else 0.5
    init_gauss_sigma = global_gocor_arguments['init_gauss_sigma_DIMP'] if 'init_gauss_sigma_DIMP' \
                                                                          in gocor_arguments_keys else 1.0
    train_label_map = global_gocor_arguments['train_label_map'] if 'train_label_map' in global_gocor_arguments else True
    v_minus_act = global_gocor_arguments['v_minus_act'] if 'v_minus_act' in gocor_arguments_keys else 'sigmoid'
    v_minus_init_factor = global_gocor_arguments['v_minus_init_factor'] if 'v_minus_init_factor' \
                                                                           in gocor_arguments_keys else 4.0

    # query loss parameters
    apply_query_loss = global_gocor_arguments[
        'apply_query_loss'] if 'apply_query_loss' in gocor_arguments_keys else False
    reg_kernel_size = global_gocor_arguments['reg_kernel_size'] if 'reg_kernel_size' in gocor_arguments_keys else 3
    reg_inter_dim = global_gocor_arguments['reg_inter_dim'] if 'reg_inter_dim' in gocor_arguments_keys else 1
    reg_output_dim = global_gocor_arguments['reg_output_dim'] if 'reg_output_dim' in gocor_arguments_keys else 1

    optimizer = global_gocor.GlobalGOCorOpt(num_iter=num_iter, init_step_length=init_step_length,
                                            init_filter_reg=init_filter_reg, steplength_reg=steplength_reg,
                                            min_filter_reg=min_filter_reg, num_dist_bins=num_dist_bins,
                                            bin_displacement=bin_displacement, init_gauss_sigma=init_gauss_sigma,
                                            v_minus_act=v_minus_act, v_minus_init_factor=v_minus_init_factor,
                                            train_label_map=train_label_map,
                                            apply_query_loss=apply_query_loss, reg_kernel_size=reg_kernel_size,
                                            reg_inter_dim=reg_inter_dim, reg_output_dim=reg_output_dim,
                                            )

    return optimizer


def define_optimizer_local_corr(local_gocor_arguments):
    """
    Defines the LocalGOCor optimizer module based on the input parameter dictionary.
    default = {'optim_iter':3, 'num_features': 512, 'search_size': 9, 'init_step_length': 1.0,
               'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5,
               'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
               'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
               'apply_query_loss': False, 'reg_kernel_size': 3, 'reg_inter_dim': 1, 'reg_output_dim': 1.0}
    """
    gocor_arguments_keys = local_gocor_arguments.keys()

    search_size = local_gocor_arguments['search_size'] if 'search_size' in gocor_arguments_keys else 9
    num_features = local_gocor_arguments['num_features'] if 'num_features' in gocor_arguments_keys else 512
    num_iter = local_gocor_arguments['optim_iter'] if 'optim_iter' in gocor_arguments_keys else 3
    init_step_length = local_gocor_arguments['init_step_length'] if 'init_step_length' in gocor_arguments_keys else 1.0
    init_filter_reg = local_gocor_arguments['init_filter_reg'] if 'init_filter_reg' in gocor_arguments_keys else 1e-2
    min_filter_reg = local_gocor_arguments['min_filter_reg'] if 'min_filter_reg' in gocor_arguments_keys else 1e-5

    # reference loss parameters
    num_dist_bins = local_gocor_arguments['num_dist_bins'] if 'num_dist_bins' in gocor_arguments_keys else 10
    bin_displacement = local_gocor_arguments['bin_displacement'] if 'bin_displacement' in gocor_arguments_keys else 0.5
    init_gauss_sigma = local_gocor_arguments['init_gauss_sigma_DIMP'] if 'init_gauss_sigma_DIMP' \
                                                                         in gocor_arguments_keys else 1.0
    v_minus_act = local_gocor_arguments['v_minus_act'] if 'v_minus_act' in gocor_arguments_keys else 'sigmoid'
    v_minus_init_factor = local_gocor_arguments['v_minus_init_factor'] if 'v_minus_init_factor' \
                                                                          in gocor_arguments_keys else 4.0

    # query loss parameters
    apply_query_loss = local_gocor_arguments[
        'apply_query_loss'] if 'apply_query_loss' in gocor_arguments_keys else False
    reg_kernel_size = local_gocor_arguments['reg_kernel_size'] if 'reg_kernel_size' in gocor_arguments_keys else 3
    reg_inter_dim = local_gocor_arguments['reg_inter_dim'] if 'reg_inter_dim' in gocor_arguments_keys else 1
    reg_output_dim = local_gocor_arguments['reg_output_dim'] if 'reg_output_dim' in gocor_arguments_keys else 1

    optimizer = local_gocor.LocalGOCorrOpt(num_iter=num_iter, search_size=search_size,
                                           init_step_length=init_step_length,
                                           init_filter_reg=init_filter_reg, min_filter_reg=min_filter_reg,
                                           num_dist_bins=num_dist_bins,
                                           bin_displacement=bin_displacement, init_gauss_sigma=init_gauss_sigma,
                                           v_minus_act=v_minus_act, v_minus_init_factor=v_minus_init_factor,
                                           apply_query_loss=apply_query_loss, reg_kernel_size=reg_kernel_size,
                                           reg_inter_dim=reg_inter_dim, reg_output_dim=reg_output_dim
                                           )
    return optimizer