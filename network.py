# 전체적인 Tacotron1 network 구성

from module import *
from text.symbols import symbols
import hyperparams as hp
import random
import torch.nn as nn
import torch


#-- Encoder Network 
class Encoder(nn.Module):
    def __init__(self, embedding_size):
    '''
    embedding_size (int) : hyperparams에는 256으로 설정되어 있음
    '''
    super(Encoder, self).__init__()
    self.embedding_size = embedding_size
    self.embed = nn.Embedding(len(symbols), embedding_size) # text embedding
    self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size ) # (input_size, hidden_size, output_size)
    self.cbhg = CBHG(hp.hidden_size)  # 나머지는 default로 지정되어 있으므로 hidden_size만 입력

    def forward(self, input_):
        # (batch_size, hidden_dim, input_dim) -> (batch_size, input_dim, hidden_dim)으로 변경
        input_ = torch.transpose(self.embed(input_),1,2) 
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)

        return memory

#-- Decoder Network
class MelDecoder(nn.Module):
    ''' decoder에는 attention과 decoder의 연결 네트워크도 들어간다.'''
    def __init__(self):
        super(MelDecoder, self).__init__()
        
        # Decoder의 prenet에는 mel-spectrogram이 들어오게 된다.
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size) 
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):
        
        # hidden state initialize
        attr_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(decoder_input.size()[0]) # batch size가 들어온다.
        outputs = list()

        # Training phase
        if self.training:
            # prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step # Decoder 반복개수 지정

            # [GO] frmae -> Decoder 가장 초기값
            prev_output = dec_input[:, :, 0]

            for i in range(timsteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden,
                    gru1_hidden = gru1_hidden,
                    gru2_hidden= gru2_hidden
                )

                outputs.append(prev_output) # Decoder의 output값들을 저장한다.
        
                # teacher_forcing -> Ground Truth를 decoder의 다음 input으로 넣어주는 기법
                if random.random() < hp.teacher_forcing_ratio:
                    prev_output = dec_input[:,:, i * hp.outputs_per_step]
                else:
                    prev_output = prev_output[:,:,-1]

            # 모든 mel-spectrogram을 concat 시킨다.
            outputs = torch.cat(outputs, 2)

        # valid phase
        else:

            # [GO] frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:,:,0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden, 
                    gru1_hidden = gru1_hidden,
                    gru2_hidden = gru2_hidden
                )

                outputs.append(prev_output)
                prev_output = prev_output[:,:,-1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        return outputs

                
# Decoder후 추가 전처리 클래스
class PostProcessingNet(nn.Module):
    '''
    CBHG를 사용하여 mel-spectrogram을 linear-spectrogram으로 변경시켜주는 역할을 한다.
    '''
    def __init__(self):
        super(PostProcessingNet,self).__init__()
        self.postcbhg = CBHG(hp.hidden_size, K=8, projection_size=hp.num_mels,
                             is_post = True)
        self.linear = SeqLinear(hp.hidden_size * 2, hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg(input_)
        out = self.linear.forward(torch.transpose(out, 1, 2)) # 출력값 의미 이해 필요!!

        return out


# Tacotron 모델 클래스 구현
class Tacotron(nn.Module):
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        memory = self.encoder.forward(characters)
        mel_output = self.decoder1.forward(mel_input, memory)  # mel_input값이 ground truth..?
        linear_output = self.decoder2.forward(mel_output)

        return mel_output, linear_output

            
        
