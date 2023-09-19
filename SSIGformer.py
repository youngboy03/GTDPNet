import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
# class transformer1(nn.Module):
#     def __int__(self, dim):
#         super(transformer1, self).__int__()
#
#
#     def forward(self,x):
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self.act = nn.SiLU(True)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x)+shortcut
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim, A,num_heads=4, qkv_bias=False,
            qk_scale=None, attn_drop=0.,
            proj_drop=0.,
            sr_ratio=1,
            linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.A=A
        self.ff=FFN(128,512,128)


        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()



            self.silu = nn.SiLU(True)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        B, N, C = x.shape#1.128.64
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#1.1.128.64


        if not self.linear:

            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#2.1.1.128.64
        else:
            raise NotImplementedError

        k, v = kv[0], kv[1]#1.1.128.64
        # self.A[self.A == 0] = float('-inf')
        # self.A[self.A == 1] = 0
        attn = (q @ k.transpose(-2, -1)) * self.scale
        zero_vec = -9e15 * torch.ones_like(attn)
        attn= torch.where(self.A>0, attn, zero_vec)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)



        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        x_in = x

        x = self._inner_attention(x)+x_in
        #x=self.ff(x)
        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.conv_q = nn.Conv1d(in_features, in_features//4, kernel_size=1, stride=1)
        self.conv_k = nn.Conv1d(in_features, in_features//4, kernel_size=1, stride=1)
        self.conv_v = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        batch, s, c = x.size()  # 2.8.4096

        x = x.permute(0, 2, 1)  # 2.4096.8

        query = self.conv_q(x)  # 2.1024.8
        key = self.conv_k(x).permute(0, 2, 1)  # 2.8.1024
        adj = F.softmax(torch.bmm(key, query), dim=-1)  # 2.8.8

        mask4 = torch.zeros(batch, s, s, device=x.device, requires_grad=False)

        index = torch.topk(adj, k=int(s / 3 * 2), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        adj = torch.where(mask4 > 0, adj, torch.full_like(adj, float('-inf')))
        adj = adj.softmax(dim=-1)

        value = self.conv_v(x).permute(0, 2, 1)  # 2.8.4096

        support = torch.matmul(value, self.weight)  # 2.8.4096
        output = torch.bmm(adj, support)  # 2.8.4096

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SEGT(nn.Module):

    def __init__(self, group, height, width):

        super(SEGT,self).__init__()

        self.group = group

        self.pool = nn.AdaptiveAvgPool2d((height//4, width//4))

        pix_num = int((height//4) * (width//4))

        self.gcn = GraphConvolution(pix_num, pix_num)

        self.conv_q=nn.Conv1d(pix_num,pix_num//4,kernel_size=1,stride=1)
        self.conv_k=nn.Conv1d(pix_num,pix_num//4,kernel_size=1,stride=1)
        self.conv_v=nn.Conv1d(pix_num,pix_num,kernel_size=1,stride=1)

    def forward(self, x):

        batch, c, h, w = x.size()#2,64,128,128
        initial = x

        x = self.pool(x)#2,64,36.36

        length = int(c / self.group)#8

        xpre = torch.zeros(batch, self.group, (h//4) * (w//4)).cuda()#2.8.256

        start = 0

        end = start + length

        for i in range(self.group):

            tmp=x[:,start:end,:,:]#4.8.36.36

            xpre[:,i,:]=torch.mean(tmp,dim=1).reshape(batch,-1)
            #x1=xpre[:,i,:]2.4096

            start = end

            end = start + length

        xpos = self.gcn(xpre).permute(0,2,1)#2.4096.8

        query = self.conv_q(xpos)#2.1024.8
        key = self.conv_k(x.reshape(batch,c,-1).permute(0, 2, 1)).permute(0, 2, 1)
        atten = F.softmax(torch.bmm(key, query), dim=-1)
        value = self.conv_v(xpos).permute(0, 2, 1)

        tu = torch.bmm(atten, value).permute(0, 2, 1).reshape(batch, c, h // 4, w // 4)
        tu = tu.reshape(batch, c, (h // 4) * (w // 4))
        initial = initial.reshape(batch, c, h * w)
        tu = torch.bmm(tu, (torch.bmm(tu.permute(0, 2, 1), initial)))
        tu = tu.reshape(batch, c, h, w)

        return tu

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BNReLU, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(16,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        return x




class SSIGformer(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, c1,
                 h1, w1,
                 model='normal'):
        super(SSIGformer, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.pat=A.shape[0]
        self.model = model
        self.Q_col_norm = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.Q_row_norm = Q / (torch.sum(Q, 1, keepdim=True))  # 列归一化Q

        layers_count = 2

        #denoise
        self.CNNSA = nn.Sequential()

        self.CNNSA.add_module('CNN_SA_BN1' , nn.BatchNorm2d(self.channel))
        self.CNNSA.add_module('CNN_SA_Conv1' , nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
        self.CNNSA.add_module('CNN_SA_Act1', nn.LeakyReLU())

        self.CNNSA.add_module('CNN_SA_BN2' , nn.BatchNorm2d(128), )
        self.CNNSA.add_module('CNN_SA_Conv2', nn.Conv2d(128, 128, kernel_size=(1, 1)))
        self.CNNSA.add_module('CNN_SA_Act2', nn.LeakyReLU())


        self.CNN_Branch1 = nn.Sequential()

        self.CNN_Branch1.add_module('CNN_Branch01', SSConv(128, 128, kernel_size=5))

        self.CNN_Branch2 = nn.Sequential()

        self.CNN_Branch2.add_module('CNN_Branch02', SSConv(128, 128, kernel_size=5))


        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(270, self.class_count))

        self.model1 = Attention(128, self.A)
        self.model2 = Attention(128, self.A)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        self.act1 = nn.SiLU(True)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        self.act2 = nn.SiLU(True)


        self.gamma1 = nn.Parameter(torch.ones(1))

        self.cls_se = nn.Sequential(
            Conv3x3BNReLU(64, 64 // 2),
            nn.Conv2d(64 // 2, class_count, kernel_size=1, stride=1),
        )

        self.shallow = nn.Sequential(
            Conv3x3BNReLU(c1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(64, 128),
            Conv3x3BNReLU(128, 64),
        )

        self.segt = SEGT(8, h1 // 2, w1 // 2)

        self.segt1 = SEGT(8, h1 // 2, w1 // 2)


    def forward(self, x, xspe):

        (h, w, c) = x.shape

        B, C, H, W = xspe.shape
        x1x = self.shallow(xspe).to(device)  # 1.64.72.72
        se = self.segt(x1x)
        se=self.segt1(se)


        # se=torch.squeeze(se, 0).permute([1, 2, 0]).reshape([h * w, -1])
        se = F.interpolate(self.cls_se(se), size=(h, w), mode='bilinear', align_corners=True)
        se = torch.mean(se, dim=0).permute([1, 2, 0]).reshape([h * w, -1])


        # 先去除噪声
        x_noise = self.CNNSA(torch.unsqueeze(x.permute([2, 0, 1]), 0))#1.128.145.145
        x_noise = torch.squeeze(x_noise, 0).permute([1, 2, 0])#145.145.128
        x_clean = x_noise # 直连#145.145.128

        x_clean_flatten = x_clean.reshape([h * w, -1])#21025.128

        superpixels_flatten = torch.unsqueeze(torch.mm(self.norm_col_Q.t(), x_clean_flatten),0)#1.196.128
        superpixels_flatten=superpixels_flatten.permute([0, 2, 1])
        hx = x_clean#145.145.128


        CNN_result01 = self.CNN_Branch1(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  #1.128.145.145 spectral-spatial convolution

        CNN_result1= torch.squeeze(CNN_result01, 0).permute([1, 2, 0]).reshape([h * w, -1])#1.21025.128
        CNN_result1=torch.unsqueeze(torch.mm(self.Q_col_norm.t(), CNN_result1),0)#1,196.128



        CNN_result12=self.CNN_Branch2(CNN_result01)#1.128.145.145

        CNN_result2 = torch.squeeze(CNN_result12, 0).permute([1, 2, 0]).reshape([h * w, -1])  # 1.21025.128
        CNN_result2 = torch.unsqueeze(torch.mm(self.Q_col_norm.t(), CNN_result2),0)#1.196.128



        CNN_result = torch.squeeze(CNN_result12, 0).permute([1, 2, 0]).reshape([h * w, -1])#21025.128


        H = superpixels_flatten.permute([0, 2, 1])  # 1.196.128

        H1=self.model1(H)#1.196.128

        H1=self.act1(self.fc1(torch.concat([H1,CNN_result1],dim=-1)))


        H2 = self.model2(H1)

        H2 = self.act2(self.fc2(torch.concat([H2, CNN_result2], dim=-1)))




        transformer_result = torch.squeeze(torch.matmul(self.Q_row_norm, H2),0)

        #融合方式
        # Y = 0.95*transformer_result+0.05*CNN_result
        
        Y = torch.cat([CNN_result, transformer_result,se], dim=-1)

        P = self.Softmax_linear(Y)
        Y = F.softmax(P, -1)
        return Y,P

