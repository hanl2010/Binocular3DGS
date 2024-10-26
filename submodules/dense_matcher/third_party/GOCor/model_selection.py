from models.GLUNet.GLU_Net import GLUNet_model
from models.PWCNet.pwc_net import PWCNet_model
import os.path as osp
import torch
import os


def load_network(net, checkpoint_path=None, **kwargs):
    """Loads a network checkpoint file.
    args:
        net: network architecture
        checkpoint_path
    outputs:
        net: loaded network
    """

    if not os.path.isfile(checkpoint_path):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_path))

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    try:
        net.load_state_dict(checkpoint_dict['state_dict'])
    except:
        net.load_state_dict(checkpoint_dict)
    return net


model_type = ['GLUNet', 'GLUNet_GOCor', 'PWCNet', 'PWCNet_GOCor']
pre_trained_model_types = ['static', 'dynamic', 'chairs_things', 'chairs_things_ft_sintel']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_model(model_name, pre_trained_model_type, global_optim_iter, local_optim_iter,
                 path_to_pre_trained_models='../pre_trained_models/ours/PDCNet'):
    """
    Select, construct and load model
    args:
        model_name
        pre_trained_model_type
        global_optim_iter
        local_optim_iter
        path_to_pre_trained_models
    output:
        network: constructed and loaded network
    """

    print('Model: {}\nPre-trained-model: {}'.format(model_name, pre_trained_model_type))
    if model_name not in model_type:
        raise ValueError(
            'The model that you chose does not exist, you chose {}'.format(model_name))

    if pre_trained_model_type not in pre_trained_model_types:
        raise ValueError(
            'The pre trained model that you chose does not exist, you chose {}'.format(pre_trained_model_types))

    if model_name == 'GLUNet':
        # GLU-Net uses a global feature correlation layer followed by a cyclic consistency post-processing.
        # local cost volumes are computed by feature correlation layers
        network = GLUNet_model(iterative_refinement=True, global_corr_type='feature_corr_layer',
                               normalize='relu_l2norm', cyclic_consistency=True,
                               local_corr_type='feature_corr_layer')

    elif model_name == 'GLUNet_GOCor':
        '''
        Default for global and local gocor arguments:
        global_gocor_arguments = {'optim_iter':3, 'num_features': 512, 'init_step_length': 1.0, 
                                  'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5,
                                  'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                  'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                  'apply_query_loss': False, 'reg_kernel_size': 3, 
                                  'reg_inter_dim': 1, 'reg_output_dim': 1.0}
        
        local_gocor_arguments= {'optim_iter':3, 'num_features': 512, 'search_size': 9, 'init_step_length': 1.0,
                                'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5,
                                'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                'apply_query_loss': False, 'reg_kernel_size': 3, 
                                'reg_inter_dim': 1, 'reg_output_dim': 1.0}
        '''
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {'optim_iter': global_optim_iter, 'apply_query_loss': True,
                                  'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter}
        network = GLUNet_model(iterative_refinement=True, global_corr_type='GlobalGOCor',
                               global_gocor_arguments=global_gocor_arguments, normalize='leakyrelu',
                               local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                               same_local_corr_at_all_levels=True)

    elif model_name == 'PWCNet':
        # PWC-Net uses a feature correlation layer at each pyramid level
        network = PWCNet_model(local_corr_type='feature_corr_layer')
    elif model_name == 'PWCNet_GOCor':
        local_gocor_arguments = {'optim_iter': local_optim_iter}
        # We instead replace the feature correlation layers by Local GOCor modules
        network = PWCNet_model(local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                               same_local_corr_at_all_levels=False)
    else:
        raise NotImplementedError('the model that you chose does not exist: {}'.format(model_name))

    checkpoint_fname = osp.join(path_to_pre_trained_models, model_name + '_{}'.format(pre_trained_model_type)
                                + '.pth')
    if not os.path.exists(checkpoint_fname):
        checkpoint_fname = checkpoint_fname + '.tar'
        if not os.path.exists(checkpoint_fname):
            raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_fname))

    network = load_network(network, checkpoint_path=checkpoint_fname)
    network.eval()
    network = network.to(device)

    '''
    to plot GOCor weights
    if model_name == 'GLUNet_GOCor':
        network.corr.corr_module.filter_optimizer._plot_weights(save_dir='evaluation/')
        network.local_corr.filter_optimizer._plot_weights(save_dir='evaluation/')
    '''
    return network