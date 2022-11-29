import torch
from torch import nn
from torch.hub import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )
        
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, use_residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        if use_residual:
            self.use_res_connect = self.stride == 1 and inp == oup
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
            #nn.ReLU6(inplace=True)
        ])

        
        self.conv = nn.Sequential(*layers)
        

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ThinBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, use_residual=True):
        super(ThinBlock, self).__init__()
        self.stride = stride
        
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        if use_residual:
            self.use_res_connect = self.stride == 1 and inp == oup
        else:
            self.use_res_connect = False

        layers = []
        #if expand_ratio != 1:
            # pw
            #layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        #layers.append(ConvBNReLU(inp, inp, kernel_size=1, groups=inp, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(inp, inp, stride=stride, groups=inp, norm_layer=norm_layer),
            # pw-linear
            #nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),

            norm_layer(oup),
            nn.ReLU6(inplace=True)
        ])

        
        self.conv = nn.Sequential(*layers)
        '''
        self.scale_param = nn.Parameter(torch.randn(oup))
        self.bias_param = nn.Parameter(torch.randn(oup))
        
        nn.init.ones_(self.scale_param)
        nn.init.zeros_(self.bias_param)
        '''

        #nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out')

    def forward(self, x, base):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        #base = nn.functional.adaptive_avg_pool2d(base, 1).reshape(base.shape[0], -1, 1, 1)
        #print("baseshape: {}".format(base.shape))
        #print("outshape: {}".format(out.shape))
        out = (0.1 * out) +  base
        #nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #out = out * self.scale_param.view(-1, 1, 1) + self.bias_param.view(-1, 1, 1)

        return out

class ThinBlock2(nn.Module):
    def __init__(self, inp, oup, stride, norm_layer=None, use_residual=True):
        super(ThinBlock2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        
        if use_residual:
            self.use_res_connect = self.stride == 1 and inp == oup
        else:
            self.use_res_connect = False

        layers = []
        
        layers.extend([
            ConvBNReLU(inp, inp, stride=stride, groups=inp, norm_layer=norm_layer),
            ConvBNReLU(inp, oup, stride=1, groups=1, norm_layer=norm_layer)
            # pw-linear
            #nn.Conv2d(inp, oup, 1, stride, 1, bias=False),
            #norm_layer(oup),
        ])

        
        self.conv = nn.Sequential(*layers)

        #self.scale_param = nn.Parameter(torch.randn(oup))
        #self.bias_param = nn.Parameter(torch.randn(oup))
        
        #nn.init.ones_(self.scale_param)
        #nn.init.zeros_(self.bias_param)
        

    def forward(self, x):
        #if self.use_res_connect:
        #    out = x + self.conv(x)
        #else:
        out = self.conv(x)

        #out = out + base
        #out = out * self.scale_param.view(-1, 1, 1) + self.bias_param.view(-1, 1, 1)

        return out 
    
'''
class ThinNetwork(nn.Module):
    def __init__(self, start_channel, ):
        super(ThinNetwork, init).__init()__

        features = [ConvBNReLU(3, start_channel, stride=2, norm_layer=norm_layer)]

        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            
'''     

class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        thin_block = ThinBlock2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.thin_start_layer = 20
        self.thin_end_layer = 19
        self.thin_output_channel = 512
        layer_index = 0
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        features[0].register_forward_hook(self.hookForward)
        self.thin_features = nn.ModuleList()
        self.features_output = []
        self.scale_list = []
        self.bias_list = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                feat = block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)
                feat.register_forward_hook(self.hookForward)
                features.append(feat)

                
                if layer_index >= self.thin_start_layer and layer_index <= self.thin_end_layer:
                    thin_feat = thin_block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)
                    print("Thin input: {}    Thin output channel: {}".format(input_channel, output_channel))
                    #thin_feat.register_forward_hook(self.hookForward)
                    #if self.thin_features == None:
                    #    self.thin_features = nn.ModuleList(thin_feat)
                    #else:
                    self.thin_features.append(thin_feat)
                    self.thin_output_channel = output_channel
                    #self.thin_features.append(thin_feat)

                    '''
                    self.scale_param = nn.Parameter(torch.randn(output_channel))
                    self.bias_param = nn.Parameter(torch.randn(output_channel))
        
                    nn.init.ones_(self.scale_param)
                    nn.init.zeros_(self.bias_param)
                    '''
                
                input_channel = output_channel
                layer_index += 1

        
        print("input channel: {}".format(input_channel))
        self.prethin = nn.Sequential(ConvBNReLU(3, 32, stride=2, norm_layer=norm_layer))
        self.thin_output_channel = 320
        
        '''
        self.thin_features.append(thin_block(32, 16, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(16+16, 24, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(24+24, 32, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(32+32, 64, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(64+64, 96, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(96+96, 160, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(160+160, self.thin_output_channel, stride=1, norm_layer=norm_layer))
        '''

        self.thin_features.append(thin_block(32, 8, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(16+8, 16, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(24+16, 24, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(32+24, 32, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(64+32, 64, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(96+64, 96, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(160+96, self.thin_output_channel, stride=1, norm_layer=norm_layer))


        #self.adding_list = [1, 3, 6, 10, 13, 16]
        self.adding_list = [1, 3, 6, 10, 13, 16]

        
        # building last several layers
        #features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        #post_feature = [ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)]
        #self.post_feature = nn.Sequential(*post_feature)


        #small_post = [ConvBNReLU(input_channel, self.thin_output_channel, kernel_size=1, norm_layer=norm_layer)]
        #self.small_post = nn.Sequential(*small_post)


        real_last_channel = output_channel + self.thin_output_channel
        print(real_last_channel)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(real_last_channel, num_classes),
        )

        

        self.avg_pool_base = nn.AvgPool2d(2, stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        #self.register_forward_hook(hookForward)

    def initBN(self):

        for feature in self.features:
            if "adapter.0.conv.0.weight" in feature.state_dict().keys():
                feature.adapter[0].conv[0].weight = nn.Parameter(feature.inter_conv[0].weight.clone(), requires_grad = False)

    def _forward_impl(self, x):
        self.features_output = []

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        org_input = x
        x = self.features(x)
        #p1 = self.post_feature(x)
        #p2 = self.small_post(x)


        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        #p1 = nn.functional.adaptive_avg_pool2d(p1, 1).reshape(p1.shape[0], -1)

        #p2 = nn.functional.adaptive_avg_pool2d(p2, 1).reshape(p2.shape[0], -1)

        #x = torch.cat((p1, p2), 1)

        #print(p1.shape)


        #thin_out = self.features_output[self.thin_start_layer - 1]
        #thin_out = self.avg_pool_base(thin_out)
        #thin_out = self.prethin(org_input)
        thin_out = self.features_output[0]


        #print("thin feature length: {}".format(len(self.thin_features)))
        for index, thin_layer in enumerate(self.thin_features):
            #print("thin shape: {}".format(thin_out.shape))
            if index < (len(self.thin_features) - 1) and self.adding_list[index] > 0:
                temp1 = thin_layer(thin_out)
                temp2 = self.features_output[self.adding_list[index]]
                #print(temp1.shape)
                #print(temp2.shape)
                thin_out = torch.cat((temp1, temp2), 1)
            else:
                thin_out = thin_layer(thin_out) 

        #print("thin shape: {}".format(thin_out.shape))
        thin_out = nn.functional.adaptive_avg_pool2d(thin_out, 1).reshape(thin_out.shape[0], -1)
        
        total_out = torch.cat((x, thin_out), 1)
        
        #print(total_out.shape)
        total_out = self.classifier(total_out)
        

        #total_out = self.classifier(x)

        #print(self.features_output[10].shape)
        return total_out
    
    def copyBN(self, feature_index_list):
        
        copy_index = 0

        for feature_index, feature in enumerate(self.features):
            print("feature")
            if feature_index_list[copy_index] < 0:
                print("ggantinue")
                copy_index+=1
                continue

            if feature_index == feature_index_list[copy_index]:
                print("stars")
                
                if "conv.3" in feature.state_dict().keys():

                    self.thin_features[copy_index].conv[1][1].weight = nn.Parameter(feature.conv[3].weight.clone(), requires_grad = True)
                    self.thin_features[copy_index].conv[1][1].bias = nn.Parameter(feature.conv[3].bias.clone(), requires_grad = True)

                    #self.thin_features[copy_index].conv[0][1].weight = nn.Parameter(feature.conv[1][1].weight.clone(), requires_grad = True)
                    #self.thin_features[copy_index].conv[0][1].bias = nn.Parameter(feature.conv[1][1].bias.clone(), requires_grad = True)

                    #feature.point[1].running_mean = nn.Parameter(feature.conv[0][1].running_mean.clone(), requires_grad = False)
                    #feature.point[1].running_var = nn.Parameter(feature.conv[0][1].running_var.clone(), requires_grad = False)
                elif "conv.2" in feature.state_dict().keys():
                    print(feature.conv[2])
                    self.thin_features[copy_index].conv[1][1].weight = nn.Parameter(feature.conv[2].weight.clone(), requires_grad = True)
                    self.thin_features[copy_index].conv[1][1].bias = nn.Parameter(feature.conv[2].bias.clone(), requires_grad = True)
                    #self.thin_features[copy_index].conv[0][1].weight = nn.Parameter(feature.conv[0][1].weight.clone(), requires_grad = True)
                    #self.thin_features[copy_index].conv[0][1].bias = nn.Parameter(feature.conv[0][1].bias.clone(), requires_grad = True)



                copy_index+=1
            
            if copy_index > 6:
                break
            



    def forward(self, x):
        return self._forward_impl(x)

    def hookForward(self, module, inputs, outputs):
        self.features_output.append(outputs)
        #print("output shape: {}:".format(outputs.shape))

    

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



