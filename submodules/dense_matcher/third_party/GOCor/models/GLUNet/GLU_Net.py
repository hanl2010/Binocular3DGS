import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.mod import deconv, unnormalise_and_convert_mapping_to_flow
from .base_matching_net import BaseGLUMultiScaleMatchingNet
from .base_matching_net import set_glunet_parameters
from ..feature_backbones.VGG_features import VGGPyramid
from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow, convert_flow_to_mapping
from GOCor.local_correlation import correlation


class GLUNet_model(BaseGLUMultiScaleMatchingNet):
    """
    GLU-Net model
    """
    def __init__(self, iterative_refinement=False,
                 global_corr_type='feature_corr_layer', global_gocor_arguments=None, normalize='relu_l2norm',
                 normalize_features=True, cyclic_consistency=False,
                 local_corr_type='feature_corr_layer', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 decoder_inputs='corr_flow_feat', refinement_at_adaptive_reso=True, refinement_at_all_levels=False,
                 refinement_at_finest_level=True, apply_refinement_finest_resolution=True,
                 batch_norm=True, pyramid_type='VGG', md=4, upfeat_channels=2, train_features=False):
        params = set_glunet_parameters(global_corr_type=global_corr_type, gocor_global_arguments=global_gocor_arguments,
                                       normalize=normalize, normalize_features=normalize_features,
                                       cyclic_consistency=cyclic_consistency, md=md,
                                       local_corr_type=local_corr_type, gocor_local_arguments=local_gocor_arguments,
                                       same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                                       local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                                       decoder_inputs=decoder_inputs,
                                       refinement_at_adaptive_reso=refinement_at_adaptive_reso,
                                       refinement_at_all_levels=refinement_at_all_levels,
                                       refinement_at_finest_level=refinement_at_finest_level,
                                       apply_refinement_finest_resolution=apply_refinement_finest_resolution,
                                       batch_norm=batch_norm, nbr_upfeat_channels=upfeat_channels)
        super().__init__(params)
        self.iterative_refinement = iterative_refinement

        # level 4, 16x16
        nd = 16*16  # global correlation
        od = nd + 2
        decoder4, num_channels_last_conv = self.initialize_mapping_decoder(self.params.global_decoder_type, in_channels=od,
                                                                           batch_norm=self.params.batch_norm)
        self.decoder4 = decoder4
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 3, 32x32
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        decoder3, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder3 = decoder3
        input_to_refinement_3 = num_channels_last_conv

        if self.params.give_flow_to_refinement_module:
            input_to_refinement_3 += 2
        # weights for refinement module
        if self.params.refinement_at_adaptive_reso:
            self.initialize_adaptive_reso_refinement_module(input_to_refinement_3, self.params.batch_norm)

        # level 2, 1/8 of original resolution
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        decoder2, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder2 = decoder2
        input_to_refinement_2 = num_channels_last_conv

        if 'feat' in self.params.decoder_inputs:
            self.upfeat2 = deconv(input_to_refinement_2, self.params.nbr_upfeat_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 1, 1/4 of original resolution
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        decoder1, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        in_channels_corr=nd,
                                                                        nbr_upfeat_channels=self.params.nbr_upfeat_channels)
        self.decoder1 = decoder1
        input_to_refinement_1 = num_channels_last_conv
        if self.params.give_flow_to_refinement_module:
            input_to_refinement_1 += 2
        # weights of the final refinement (context network)
        if self.params.refinement_at_finest_level:
            self.initialize_last_level_refinement_module(input_to_refinement_1, self.params.batch_norm)

        # initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1).
                # So we use the same initialization here.
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

        # initialize the global and local correlation modules
        # comprises GOCor or feature correlation layer
        self.initialize_global_corr()
        self.initialize_local_corr()

        # features back-bone extractor
        if pyramid_type == 'VGG':
            feature_extractor = VGGPyramid(train=train_features)
        else:
            raise NotImplementedError('The feature extractor that you selected in not implemented: {}'
                                      .format(pyramid_type))
        self.pyramid = feature_extractor

    def forward(self, im_target, im_source, im_target_256, im_source_256, im_target_pyr=None, im_source_pyr=None,
                im_target_pyt_256=None, im_source_pyr_256=None):
        # im1 is target image, im2 is source image
        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = 1.0

        c14, c24, c13, c23, c12, c22, c11, c21 = self.extract_features(im_target, im_source, im_target_256,
                                                                       im_source_256, im_target_pyr,
                                                                       im_source_pyr, im_target_pyt_256,
                                                                       im_source_pyr_256)
        # RESOLUTION 256x256
        # level 4: 16x16
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = self.deconv4(flow4)

        # level 3: 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)
        # constrained correlation now
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr3 = self.local_corr(c13, warp3)
            else:

                corr3 = self.local_corr_3(c13, warp3)
        else:
            # feature correlation layer
            corr3 = correlation.FunctionCorrelation(c13, warp3)
        corr3 = self.leakyRELU(corr3)

        if self.params.decoder_inputs == 'corr_flow_feat':
            corr3 = torch.cat((corr3, up_flow4), 1)
        elif self.params.decoder_inputs == 'corr':
            corr3 = corr3
        elif self.params.decoder_inputs == 'corr_flow':
            corr3 = torch.cat((corr3, up_flow4), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))
        x3, res_flow3 = self.decoder3(corr3)

        if self.params.refinement_at_adaptive_reso:
            if self.params.give_flow_to_refinement_module:
                input_refinement = res_flow3 + up_flow4
                x3 = torch.cat((x3, input_refinement), 1)
            x_, res_flow3_ = self.PWCNetRefinementAdaptiveReso(x3)
            res_flow3 = res_flow3 + res_flow3_

        flow3 = res_flow3 + up_flow4

        if self.iterative_refinement:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_original)/8.0/32.0
            R_h = float(w_original)/8.0/32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R/minimum_ratio)/np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n ))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_original * ratio), int(w_original * ratio)),
                                             mode='bilinear',
                                             align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_original * ratio),
                                                                         int(w_original * ratio)), mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_original * ratio),
                                                                         int(w_original * ratio)), mode='area')
                    warp3 = self.warp(c23_bis, up_flow3 * div * ratio)

                    if 'GOCor' in self.params.local_corr_type:
                        if self.params.same_local_corr_at_all_levels:
                            corr3 = self.local_corr(c13_bis, warp3)
                        else:

                            corr3 = self.local_corr_3(c13_bis, warp3)
                    else:
                        # feature correlation layer
                        corr3 = correlation.FunctionCorrelation(c13_bis, warp3)
                    corr3 = self.leakyRELU(corr3)

                    if self.params.decoder_inputs == 'corr_flow_feat':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    elif self.params.decoder_inputs == 'corr':
                        corr3 = corr3
                    elif self.params.decoder_inputs == 'corr_flow':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_original) / float(256)
            up_flow3[:, 1, :, :] *= float(h_original) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # level 2 : 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = self.warp(c22, up_flow3*div*ratio)
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr2 = self.local_corr(c12, warp2)
            else:

                corr2 = self.local_corr_2(c12, warp2)
        else:
            # feature correlation layer
            corr2 = correlation.FunctionCorrelation(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        if self.params.decoder_inputs == 'corr_flow_feat':
            corr2 = torch.cat((corr2, up_flow3), 1)
        elif self.params.decoder_inputs == 'corr':
            corr2 = corr2
        elif self.params.decoder_inputs == 'corr_flow':
            corr2 = torch.cat((corr2, up_flow3), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))

        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = self.deconv2(flow2)
        if self.params.decoder_inputs == 'corr_flow_feat':
            up_feat2 = self.upfeat2(x2)

        # level 1: 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = self.warp(c21, up_flow2*div*ratio)
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr1 = self.local_corr(c11, warp1)
            else:

                corr1 = self.local_corr_1(c11, warp1)
        else:
            # feature correlation layer
            corr1 = correlation.FunctionCorrelation(c11, warp1)
        corr1 = self.leakyRELU(corr1)

        if self.params.decoder_inputs == 'corr_flow_feat':
            corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        elif self.params.decoder_inputs == 'corr':
            corr1 = corr1
        elif self.params.decoder_inputs == 'corr_flow':
            corr1 = torch.cat((corr1, up_flow2), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))

        x, res_flow1 = self.decoder1(corr1)

        if self.params.refinement_at_finest_level and self.params.apply_refinement_finest_resolution:
            if self.params.give_flow_to_refinement_module:
                input_refinement = res_flow1 + up_flow2
                x = torch.cat((x, input_refinement), 1)
            x_, res_flow1_ = self.PWCNetRefinementFinal(x)
            res_flow1 = res_flow1 + res_flow1_

        flow1 = res_flow1 + up_flow2

        # prepare output dict
        output = {'flow_estimates': [flow2, flow1]}
        output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4}
        return output_256, output

    # FOR FLIPPING CONDITION
    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return corr4, flow4

    def flipping_condition(self, im_source_base, im_target_base, device):

        if self.params.global_corr_type == 'GlobalGOCor':
            # flipping condition specific to the GOCor modules
            condition = 'max_corr'
        else:
            condition = 'min_avg_flow'

        list_average_flow = []
        false_true = [False, True]
        for apply_flipping in false_true:
            im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping)
            b, _, h_original, w_original = im_target.size()
            b, _, h_256, w_256 = im_target_256.size()

            # pyramid, 256 reso
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
            c14 = im1_pyr_256[-3]
            c24 = im2_pyr_256[-3]

            corr4, flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
            if condition == 'min_avg_flow':
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
            else:
                value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                average_flow = value.sum()
            list_average_flow.append(average_flow.item())
        if condition == 'min_avg_flow':
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
        else:
            target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]
        if target_image_is_flipped:
            list_average_flow = []

            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                    self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping)
                b, _, h_original, w_original = im_target.size()
                b, _, h_256, w_256 = im_target_256.size()

                # pyramid, 256 reso
                im1_pyr_256 = self.pyramid(im_target_256)
                im2_pyr_256 = self.pyramid(im_source_256)
                c14 = im1_pyr_256[-3]
                c24 = im2_pyr_256[-3]

                corr4, flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                if condition == 'min_avg_flow':
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                else:
                    value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                    average_flow = value.sum()
                list_average_flow.append(average_flow.item())
            if condition == 'min_avg_flow':
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            else:
                target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]

        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x_original, ratio_y_original = \
            self.pre_process_data(im_source_base, im_target_base, apply_flip=target_image_is_flipped)
        return im_source.to(device).contiguous(), im_target.to(device).contiguous(), \
               im_source_256.to(device).contiguous(), im_target_256.to(device).contiguous(), \
               ratio_x_original, ratio_y_original

    def estimate_flow_with_flipping_condition(self, source_img, target_img, output_shape=None,
                                              scaling=1.0, mode='channel_first'):
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y = \
            self.flipping_condition(source_img, target_img, self.device)

        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        if self.target_image_is_flipped:
            flipped_mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)\
                .permute(0, 2, 3, 1).cpu().numpy()
            b = flipped_mapping.shape[0]
            mapping_per_batch = []
            for i in range(b):
                map = np.copy(np.fliplr(flipped_mapping[i]))
                mapping_per_batch.append(map)

            mapping = torch.from_numpy(np.float32(mapping_per_batch)).permute(0, 3, 1, 2).to(self.device)
            flow_est = convert_mapping_to_flow(mapping, self.device)

        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)