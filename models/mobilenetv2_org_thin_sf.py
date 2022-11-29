import torch
from torch import nn
from torch.hub import load_state_dict_from_url

#from .dynamic_conv import Dynamic_conv2d

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes)
        )



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

class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU6(inplace=True)
        )


class AffineConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        super(AffineConv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        conv_layer = [
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        ]

        post_layer = [
                norm_layer(out_planes),
                nn.ReLU6(inplace=True)
        ]
        
        self.conv = nn.Sequential(*conv_layer)
        self.post = nn.Sequential(*post_layer)

    def forward(self, x):

        out = self.conv(x)
        out = self.post(out)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, use_point="none", out_point_dim=-1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_point = use_point
        self.out_point_dim = out_point_dim
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        pre_layers = []
        self.pre_layers = None
        if expand_ratio != 1 and use_point != "pre" and use_point != "both":
            # pw
            pre_layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            self.pre_layers = nn.Sequential(*pre_layers)


        self.middle_layer = AffineConv(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer)

        if use_point != "post" and use_point != "both":
            print(hidden_dim)
            print(oup)
            layers.extend([
                #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #norm_layer(oup),
                ConvBN(hidden_dim, oup, kernel_size=1, stride=1, norm_layer=norm_layer),
            ])
            self.conv = nn.Sequential(*layers)
        
        if out_point_dim > 0:
            #print("baby")
            point_layer = [
                nn.Conv2d(oup, out_point_dim, 1, 1, 0, groups=1, bias=False),
                norm_layer(out_point_dim),
                #nn.LeakyReLU(inplace=True)
                nn.ReLU6(inplace=True)
                #nn.PReLU(out_point_dim)

            ]

            point_layer_after = [
                nn.Conv2d(out_point_dim, oup, 1, 1, 0, groups=1, bias=False),
                norm_layer(oup),
                #nn.Sigmoid()
                #nn.LeakyReLU(inplace=True)
                #nn.PReLU(oup)
                #nn.LeakyReLU(inplace=True)
                nn.ReLU6(inplace=True)
            ]
            
            self.out_point_conv = nn.Sequential(*point_layer)
            self.out_point_after_conv = nn.Sequential(*point_layer_after)
        

    def forward(self, x, train_ac=False):

        
        out = x
        aux_out = None

        #print(type(out))
        
        if self.pre_layers is not None and self.use_point != "pre" and self.use_point != "both":
            out = self.pre_layers(out)
        
        out = self.middle_layer(out)
        middle_out = out

        if self.use_point != "post" and self.use_point != "both":
            if self.use_res_connect and self.use_point == "none":
                out = x + self.conv(out)
                #out = self.conv(out)

            else:
                out = self.conv(out)
            if self.out_point_dim > 0:
                ae_input = out
                aux_out = self.out_point_conv(out)
                #aux_out = out
                
                #ae_input = middle_out
                #aux_out = self.out_point_conv(middle_out)


                if train_ac:
                    aux_out = self.out_point_after_conv(aux_out)
                    

        block_out = out
        
        if self.out_point_dim > 0:
            return out, aux_out, ae_input
        else:
            return out


class ThinBlock(nn.Module):
    def __init__(self, inp, oup, stride, norm_layer=None, use_residual=True, pw_input=-1, scale=2):
        super(ThinBlock, self).__init__()
        self.stride = stride
        self.pw_input = pw_input
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        if use_residual:
            self.use_res_connect = self.stride == 1 and inp == oup
        else:
            self.use_res_connect = False

        layers = []

        scale = 4

        #conv2d = Dynamic_conv2d(inp, oup, 1, stride=stride)

        layers.extend([
            #ConvBNReLU(inp, oup, stride=1, kernel_size=1, groups=1, norm_layer=norm_layer),
            #ConvBNReLU(oup, oup, stride=stride, kernel_size=3, groups=oup, norm_layer=norm_layer),
            ConvBNReLU(inp, inp * scale, stride=1, kernel_size=1, groups=1, norm_layer=norm_layer),
            ConvBNReLU(inp * scale, inp * scale, stride=stride, kernel_size=3, groups=(inp * scale), norm_layer=norm_layer),
            ConvBNReLU(inp * scale, oup, stride=1, kernel_size=1, groups=1, norm_layer=norm_layer)
            

            # pw-linear
            #nn.Conv2d(inp, oup, 1, stride=1, 1, bias=False),
            #norm_layer(oup),
            #conv2d,
            #norm_layer(oup),
            #nn.ReLU6(inplace=True)
            ])

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        #if self.use_res_connect:
        #    out = x + self.conv(x)
        #else:
        if self.pw_input > 0:
            #out = x + self.input_point(y)
            out = self.conv(x)
        else:
            out = self.conv(x)


        return out 

class BigResidual(nn.Module):
    def __init__(self, small_residual_setting, input_channel, block, norm_layer, out_point_dim=-1):
        super(BigResidual, self).__init__()

        expansion_factor = small_residual_setting[0]
        output_channel = small_residual_setting[1]
        n = small_residual_setting[2]
        first_stride = small_residual_setting[3]
        
        use_point = "none"
        self.out_point_dim = out_point_dim

        iv_pre = [block(input_channel, output_channel, first_stride, expand_ratio=expansion_factor, norm_layer=norm_layer, use_point=use_point)]
        iv_list = []

        input_channel = output_channel
        oupd = -1

        
        for i in range(1, n):
            print("oops")
            stride = first_stride if i == 0 else 1
           
            if n > 2:
                if i == 0:
                    use_point = "none"
                elif i == 1:
                    use_point = "post"
                elif i == (n - 1):
                    use_point = "pre"
                    oupd = out_point_dim
                else:
                    use_point = "both"
            else:
                use_point = "none"

            
            iv_list.append(block(input_channel, output_channel, stride, expand_ratio=expansion_factor, norm_layer=norm_layer, use_point=use_point, out_point_dim=oupd))

            input_channel = output_channel

        self.pre_inverted = nn.Sequential(*iv_pre)
        self.inverted_residuals_list = nn.ModuleList(iv_list)
        #print(self.inverted_residuals_list)

    def forward(self, x, train_ac=False):
        
        pre = self.pre_inverted(x)
        
        out = self.inverted_residuals_list[0](pre)

        if self.out_point_dim > 0:
            out, aux_out, ae_input = self.inverted_residuals_list[1](out, train_ac)
            out = out + pre
            return out, aux_out, ae_input

        else:
            out = self.inverted_residuals_list[1](out, train_ac)
            out = out + pre
            return out


        #out = out + pre

        #return out, aux_out, ae_input
    

class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        thin_block = ThinBlock

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
        
        
        big_inverted_list = [
            #[6, 160, 3, 2],
                
        ]
        

        
        big_block = BigResidual


        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.adding_list = [16, 17] 
        #self.adding_list = [17]
        #self.adding_dim_list = [240]
        #self.adding_list = [3, 6, 10, 13, 14]
        #self.adding_dim_list = [68, 80, 100]

        self.adding_dim_list = [120, 160]
        #self.adding_dim_list = [8, 16, 24, 32, 64]
        #self.adding_dim_list = [8, 16, 24, 32, 64, 96]
        self.thin_input_index = 13


        self.thin_adding_list = [1, 2]

        self.train_ae = True

        self.thin_start_layer = 20
        self.thin_end_layer = 19
        self.thin_output_channel = 320
        layer_index = 1
        '''
        thin_adapter_setting = [
            #[32, 8, 2],
            #[24, 32, 2, 4],
            #[32, 64, 2, 2],
            #[64, 88, 1, 1],
            #[88, self.thin_output_channel, 2, 8],
            #[120, self.thin_output_channel, 1, 2],
            [160, self.thin_output_channel, 1, 2]
            #[320, self.thin_output_channel, 1]
        ]
        '''
        thin_adapter_setting = [
            #[32, 12, 1],
            #[12, 20, 2],
            #[20, 30, 2, 1],
            #[64, 88, 1, 2],
            [96, 120, 2, 4],
            [120, 160, 1, 4],
            [160, self.thin_output_channel, 1, 2]
        ]

        '''
        thin_adapter_setting = [
            [64, 80, 1],
            #[80, 96, 2],
            [80, 120, 1],
            [120, self.thin_output_channel, 1]
        ]
        '''
        

        '''
        thin_adapter_setting = [
            [160, 240, 1],
            [240, self.thin_output_channel, 1]
        ]
        '''


        '''
        thin_adapter_setting = [
            [32, 8, 2],
            [8, 16, 2],
            [16, 24, 2],
            [24, 32, 1],
            [32, 64, 2],
            [64, self.thin_output_channel, 1]
        ]
        '''

        '''
        thin_adapter_setting = [
            [32, 8, 2],
            [16, 24, 2],
            [40, 64, 2],
            [88, 128, 1],
            [160, 224, 2],
            [288, self.thin_output_channel, 1]
        ]
        '''


        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
            
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        
        self.thin_features = nn.ModuleList()
        self.features_output = []
        # building inverted residual blocks

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                print(input_channel)

                #print(layer_index)
                if layer_index in self.adding_list:
                    out_point_dim = self.adding_dim_list[self.adding_list.index(layer_index)]
                    print("input: {} output: {}, opd: {}".format(input_channel, output_channel, out_point_dim))
                else:
                    out_point_dim = -1

                stride = s if i == 0 else 1
                
                feat = block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, out_point_dim=out_point_dim)
                features.append(feat)
                                                
                input_channel = output_channel
                layer_index += 1

        for small_inverted in big_inverted_list:
            print("big")

            if layer_index in self.adding_list:
                out_point_dim = self.adding_dim_list[self.adding_list.index(layer_index)]
            else:
                out_point_dim = -1

            out_point_dim = self.adding_dim_list[self.adding_list.index(layer_index)]
            print(out_point_dim)

            feat = big_block(small_inverted, input_channel, block, norm_layer=norm_layer, out_point_dim=out_point_dim)

            features.append(feat)
            input_channel = small_inverted[1]

            layer_index += 1
        
        #output_channel = big_inverted_list[-1][1]

        
        print("input channel: {}".format(input_channel))
        #self.prethin = nn.Sequential(ConvBNReLU(3, 32, stride=2, norm_layer=norm_layer))
        #self.thin_output_channel = 160
        
        
        for in_channel, out_channel, stride, scale in thin_adapter_setting:
            self.thin_features.append(thin_block(in_channel, out_channel, stride=stride, norm_layer=norm_layer, scale=scale))
        

        '''
        #self.thin_features.append(thin_block(32, 8, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(32, 8, stride=2, norm_layer=norm_layer))

        #self.thin_features.append(thin_block(8, 16, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(8, 24, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(24, 32, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(32, 64, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(64, 96, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(96, self.thin_output_channel, stride=1, norm_layer=norm_layer, pw_input=160))
        '''

        '''
        self.thin_features.append(thin_block(32, 8, stride=2, norm_layer=norm_layer))

        for in_channel, out_channel, stride in thin_adapter_setting:
            self.thin_features.append(thin_block(in_channel, out_channel, stride=stride, norm_layer=norm_layer))

        
        self.thin_features.append(thin_block(16, 24, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(40, 64, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(88, 128, stride=1, norm_layer=norm_layer))
        self.thin_features.append(thin_block(160, 224, stride=2, norm_layer=norm_layer))
        self.thin_features.append(thin_block(288, self.thin_output_channel, stride=1, norm_layer=norm_layer, pw_input=160))
        '''
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        #last_layer = ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        print(self.last_channel)
        
        self.feature_list = nn.ModuleList(features)
        #self.last_layer = nn.Sequential(*last_layer)

        real_output_channel = self.last_channel + self.thin_output_channel
        print(real_output_channel)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(real_output_channel, num_classes),
        )

        def update_temperature(self):
            for m in self.modules():
                if isinstance(m, Dynamic_conv2d):
                    m.update_temperature()

        #self.avg_pool_base = nn.AvgPool2d(2, stride=2)

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

    


    def setTrainAE(self, train_ae):
        self.train_ae = train_ae

    def initBN(self):

        for feature in self.features:
            if "adapter.0.conv.0.weight" in feature.state_dict().keys():
                feature.adapter[0].conv[0].weight = nn.Parameter(feature.inter_conv[0].weight.clone(), requires_grad = False)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()
                                                    

    def _forward_impl(self, x):
        self.features_output = []
        point_output_list = []
        ae_input_list = []
                        
        for index, base_layer in enumerate(self.feature_list):
            #print(index)
            
            if index in self.adding_list:
                                
                x, aux_out, ae_input = base_layer(x, self.train_ae)
                point_output_list.append(aux_out)
                ae_input_list.append(ae_input)
                #print(type(aux_out))
            else:
                x = base_layer(x)

            if index == self.thin_input_index:
                first_output = x
        #print(x.shape)
        if self.train_ae:
            return point_output_list, ae_input_list

        adding_count = 0
        thin_input = first_output
        #print(thin_input.shape)
        for index, thin_layer in enumerate(self.thin_features):
            #print("thin shape: {}".format(thin_out.shape))
            if index in self.thin_adding_list:
                #print(point_output_list[adding_count].shape)
                #print(thin_input.shape)

                #thin_input = torch.cat((thin_input, point_output_list[adding_count]), 1)
                #print(thin_input.shape)
                #print(point_output_list[adding_count].shape)

                thin_input = thin_input + point_output_list[adding_count]
                adding_count += 1

            thin_input = thin_layer(thin_input)
        
        #thin_out = thin_input
        thin_out = nn.functional.adaptive_avg_pool2d(thin_input, 1).reshape(thin_input.shape[0], -1)
        #x = torch.cat((x, thin_out), 1)
        #print(x.shape)
        #print(thin_out.shape)
        #x = x + 0.1 * thin_out
        #x = self.last_layer(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        
        total_out = torch.cat((x, thin_out), 1)
        #print(total_out.shape)
        
        total_out = self.classifier(total_out)

        return total_out

    def _forward_base(self, x):
        
        for index, base_layer in enumerate(self.feature_list):
            #print(index)
            #print(self.adding_list)
            if index in self.adding_list:
                #print(index)                   
                x, aux_out, ae_input = base_layer(x, self.train_ae)
                #point_output_list.append(aux_out)
                #ae_input_list.append(ae_input)
                #print(type(aux_out))
            else:
                x = base_layer(x)
            
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        out = self.classifier(x)

        return out
    
    def forward(self, x):
        y = self._forward_impl(x)
        #y = self._forward_base(x)


        return y


def mobilenet_v2_org_thin_sf(pretrained=False, progress=True, **kwargs):
    
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



