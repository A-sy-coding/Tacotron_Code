# module.py에서는Tacotron의 구성 모듈들을 구현해보려고 한다.
import torch
import torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderDict
import numpy as np
import hyperparams as hp


'''
CBHG 
-> 1D Convolution Bank , HighWay, GRU로 구성되어 있다.

Progress:
    1D Convolution Bank -> k=16
    Max Pooling -> stride=1, width=2 : sequence에 따라 변하지 않는 부분 추출
    Conv1D projection + ReLU
    Residual Connection
    Highway (4 Layer + ReLU) :: high-level feature을 생성한다.
    Bidirectional GRU 
'''

class CBHG(nn.Module):
    '''
    Conv1d Bank, Highway, GRU
    Args:
        hidden_size (int)
        K (int) : convolution bank 개수
        projection_size(int) : 벡터 투영 사이즈
        num_gru_layers(int) 
        max_pool_kernel_size(int) : max pooling 사이즈
        is_post(boolean) : post processing or not 
    '''
    def __init__(self, hidden_size, K=16, projection_size= 128, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(
                                      in_channels = projection_size,
                                      out_channels = hidden_size,
                                      kernel_size = 1,
                                      padding = int(np.floar(1/2))
                                  ))

        # convolution bank 16개 생성 -> pytorch module에 알려준다.
        for i in range(2, K+1):
            self.convolution_list.append(nn.Conv1d(
                                             in_channels = hidden_size,
                                             out_channels = hidden_size,
                                             kernel_size = i,
                                             padding = int(np.floor(i/2))
                                         ))

        # batchnorm 16개 수행
        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnrom_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K # Conv1d output dim
        print('Conv Bank outdim 확인 : ', convbank_outdim)

       # Conv1d Projection
        if is_post:
            self.conv_projection_1 = nn.Conv1d(
                in_channels = convbank_outdim,
                out_channels = hidden_size * 2,
                kernel_size = 3,
                padding = int(np.floor(3/2))
            )
            self.conv_projection_2 = nn.Conv1d(
                in_channels = hidden_size * 2,
                out_channels = projection_size,
                kernel_size = 3,
                padding = int(np.floor(3/2))
            )
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)
        else:
           self.conv_projection_1 = nn.Conv1d(
                in_channels = convbank_outdim,
                out_channels = hidden_size,
                kernel_size = 3,
                padding = int(np.floor(3/2))
            )
            self.conv_projection_2 = nn.Conv1d(
                in_channels = hidden_size
                out_channels = projection_size,
                kernel_size = 3,
                padding = int(np.floor(3/2))
            )
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)

        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size,
                          num_layers=2, batch_first=True, bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        ''' 함수 이름 앞의 _는 로컬에서만 쓰인다는 의미를 가진다.'''
        if kernel_size % 2 ==0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):
        ''' PreNet의 출력값으로 나온 ouput값이 CBHG의 input으로 들어간다.'''
        input_ = input_.contiguos()
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_

        # ConvBank list와 Batchnorm list를 forward 진행
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = F.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input) 
    
        # 각 특징들을 concat을 시킨다.
        conv_cat = torch.cat(convbank_list, dim=1)
        
        # max-pooling 수행
        conv_cat = self.max_pool(conv_cat)[:,:,:,:-1]

        # Projection
        conv_projection = F.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # Highway Networks
        highway = self.highway.forward(conv_projection)
        highway = torch.transpose(highway, 1, 2)

        # Bidirectional GRU
        if use_cuda:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size)).cuda()
        else:
            init_gru = Variable(torch.zeros(2 * self.nun_gru_layers, batch_size, self.hidden_size))

        self.gru.flatten_parameters()
        out, _ = self.gru(highway, init_gru)

        return out

#### Highway Module
class Highwaynet(nn.Module):
    ''' HighwayNetwork '''
    def __init__(self, num_units, num_layers=4):
        '''
        Args:
            num_units : dimension of hidden unit
            num_layers : highway layers
        '''
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()

        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):
        
        out = input_  # conv_projection을 통해 나온 output값
        
        # highway gate function
        for fc1, fc2 in zip(self.linear, self.gates):
            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - tj
            out = h * t + out * c

### FC layer ㅡ-> SeqLinear
class SeqLinear(nn.Module):
    ''' FC layer'''
    def __init__(self, input_size, output_size, time_dim=2):
        '''
        Args:
            input_size : input의 차원
            output_size : output의 차원
            time_dim : time dimension의 인덱스
        '''
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        batch_size = input_.size()[0]

        if self.time_dim == 2:
            input_ = input_.transpose(1, 2).contiguos()
        intput_ = input_.view(-1, self.iput_size)

        out = self.linear(input_).view(batch_size, -1, self.output_size)

        if self.time_dim == 2:
            out = out.contiguous().transpose(1,2)

        return out

### Encoder Decoder에서 사용하는 PreNet 클래스 정의
class Prenet(nn.Module):
    ''' Prenet은 FC Layer + ReLU + DropOut으로 구성되어 있다.'''
    def __init__(self, input_size, hidden_size, output_size):
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer  = nn.Sequential(OrderDict([
                                                  ('fc1', SeqLinear(self.input_size, self.hidden_size)),
                                                  ('relu1', nn.ReLU()),
                                                  ('dropout1', nn.Dropout(0.5)),
                                                  ('fc2', SeqLinear(self.hidden_size, self.output_size)),
                                                  ('relu2', nn.ReLU()),
                                                  ('dropout2', nn.Dropout(0.5)),
                                              ]))

    def forward(self, input_):
        out = self.layer(input_)

        return out

### Attention과 Decoder간의 상호작용 클래스
class AttentionDecoder(nn.Module):
    def __init__(self, num_units):
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units # hidden unit의 dimension
        
        # Attention과 Decoder를 하기 위해 필요한 layers
        # 유사도 점수를 구하기 위한 사전 가중치 작업 -> 공식 확인!!
        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias = False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)

        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)

        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, hp.num_mels * hp.outputs_per_step)

    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden, gru2_hidden):
        '''
        Args:
            decoder_input : Decoder의 input 값
            memory : Encoder 에서 output으로 받은 hidden vector 
        '''

        memory_len = memory.size()[1]
        batch_size = memory.size()[0]

        # Key값 얻기 -> Encoder hidden vector를 이용해 구하기
        keys =  self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)

        # Decoder의 attentionRNN 사용
        d_t = self.attn_grucell(decoder_input, attn_hidden)

        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)

        # attention score와 attention 가중치 구하기
        attn_weights = self.v(F.tanh(keys + d_t_duplicate)).view(-1, self.num_units)).view(-1, memory_len, 1)  # 유사도 score 구하기
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)  # query & key를 이용하여 가중치 값 구하기
    
        # original query를 concat
        # torch.bmm은 batch까지 고려하여 행렬을 계산해준다.
        d_t_prime = torch.bmm(attn_weights.view([batch_size, 1, -1]), memory).squeeze(1)

        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1)) # attention-rnn에서 나온 hidden vector과 context vector를 concat시킨다.
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        
        gru2_input = gru1_input + gru1_hidden
        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input _ gru2_hidden

        # output
        output = self.out(bf_out).view(-1, hp.num_mels, hp.outputs_per_step)
         
        return output, d_t, gru1_hidden, gru2_hidden

    def inithidden(self, batch_size):
        ''' hidden layer들 초기값 설정'''
        if use_cuda:
            attn_hidden = Variable(torch.zeros(batch_sze, self.num_units), requires_grad=False).cuda()
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
        else:
            attn_hidden = Variable(torch.zeros(batch_sze, self.num_units), requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)

        return attn_hidden, gru1_hidden, gru2_hidden

