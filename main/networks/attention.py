import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math



class DAUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super(DAUnit, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc





def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func


class Conv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = createConvFunc(pdc)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




class DCA(nn.Module):
    def __init__(self, in_channels):
        super(DCA, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv2d_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=2,
            kernel_size = 5, 
            groups=in_channels
        )

        self.conv2d_3 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=3,
            kernel_size = 7, 
            groups=in_channels
        )
        self.conv2d_6 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=4,
            kernel_size = 9, 
            groups=in_channels
        )
        self.conv2d_9 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=5,
            kernel_size = 11, 
            groups=in_channels
        )


        self.CDC = Conv2d('cd', in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)





    def forward(self, x):
        out = self.CDC(x)

        attention = self.conv1(out) + self.conv2d_1(out) + self.conv2d_3(out) + self.conv2d_6(out) + self.conv2d_9(out)



        return x*attention  


class EDCA(nn.Module):
    def __init__(self, in_channels):
        super(EDCA, self).__init__()
        self.CDC = DAUnit(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//8, bias=False)
        self.DCDC = DAUnit(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels//8, bias=False)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        self.conv2d_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=2,
            kernel_size = 5, 
            groups=in_channels
        )

        self.conv2d_3 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=3,
            kernel_size = 7, 
            groups=in_channels
        )
        self.conv2d_6 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=4,
            kernel_size = 9, 
            groups=in_channels
        )
        self.conv2d_9 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            padding=5,
            kernel_size = 11, 
            groups=in_channels
        )


    def forward(self, x):

        # Local Detail Perception
        attention = self.DCDC(x) + self.CDC(x)

        # Multi-scale Artifact Perception
        attention = self.conv1(attention) + self.conv2d_1(attention) + self.conv2d_3(attention) + self.conv2d_6(attention) + self.conv2d_9(attention)



        return x*attention  




class HighFreqLearning(nn.Module):
    def __init__(self, in_channels):
        super(HighFreqLearning, self).__init__()

        self.rgb_conv1 = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.rgb_conv2 = nn.Conv2d(in_channels//8, in_channels, kernel_size=1)

        self.d_conv1 = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.d_conv2 = nn.Conv2d(in_channels//8, in_channels, kernel_size=1)

        self.dca = DCA(in_channels//8)
        
        self.edca = EDCA(in_channels//8)

        self.EGF = XModalChannelWiseTransformerBlock(in_channels)
    def forward(self, x1, x2):
        
        y1 = self.rgb_conv1(x1)
        y1 = self.edca(y1)

        y2 = self.d_conv1(x2)
        y2 = self.dca(y2)
        

        y1 = self.rgb_conv2(y1) + x1
        y2 = self.d_conv2(y2) + x2

        return self.EGF(y2, y1), y2




class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class XModalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, channel_dim=256):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.channel_dim = channel_dim
        self.attend = nn.Softmax(dim=-1)

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.down = nn.Linear(channel_dim, 1)

    def forward(self, q, k):
        q = self.to_q(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        k = self.to_k(k) 
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2))

        attn = self.attend(dots).view(dots.size(0), self.channel_dim, -1)
        attn = self.down(attn)
        return attn


class XModalChannelWiseTransformerBlock(nn.Module):
    def __init__(self, num_patches, patch_dim=1, dim=64, heads=1, dim_head=64, dropout=0.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(patch_dim)
        self.projection = nn.Linear(patch_dim ** 2, dim)
        self.projection_k = nn.Linear(patch_dim ** 2, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.mha = XModalAttention(dim, heads=heads, dim_head=dim_head, channel_dim=num_patches)

    def forward(self, q, v):

        q = self.avg_pool(q)
        k = self.avg_pool(v)
        q = q.flatten(-2)
        k = k.flatten(-2)

        q = self.projection(q)
        q += self.pos_embedding

        k = self.projection_k(k)
        k += self.pos_embedding 

        attn = self.mha(q, k).unsqueeze(-1)

        return v*attn



class ChannelWiseTransformerBlock(nn.Module):
    def __init__(self, num_patches, patch_dim=1, dim=64, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(patch_dim)
        self.projection = nn.Linear(patch_dim ** 2, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.GradEnh = DAUnit(num_patches, num_patches, kernel_size=3, padding=1, groups=num_patches, bias=False)

        self.MHT = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        res = self.GradEnh(z) 
        x = self.avg_pool(z)
        x = x.flatten(-2)

        x = self.projection(x)
        x += self.pos_embedding

        x = self.MHT(x)
        x = x.mean(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)

        return res * x

















