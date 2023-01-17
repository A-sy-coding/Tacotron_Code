from network import *
from data import get_dataset, DataLoader, collate_fn, get_param_size 
from torch import optim
import numpy as np
import argparse
import os, time
import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available() # cuda 사용 가능 여부

def main(args):
    
    # data load
    dataset = get_dataset()
    
    # use cpu or gpu
    if use_cuda:
        print('활용 가능한 gpu 개수 : ', torch.cuda.device_count(), '개')
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = hp.lr)

    # Make checkpoint
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)


    # loss function
    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    
    # loss for frequency of human register
    n_priority_freq = int(3000 / hp.sample_rate * 0.5) * hp.num_freq)

    for epoch in range(hp.epochs):
        # dataloader -> cpu에서 저장한 데이터를 gpu로 불러올 때 병목현상이 일어나는 것을
        # 방지하게 위해 num_workers라는 옵션을 준다.
        dataloader = DataLoader(dataset, batch_size = args.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True,
                                num_workers=4)

        # batch size 만큼 load한 데이터 학습 진행
        for i, data in enumerate(dataloader):
            current_step = i + args.restore_step + epoch + len(dataloader) + 1 

            optimizer.zero_grad()

            # Make decoder input
            try:
                mel_input = np.concatenate((np.zeros([args.batch_size, hp.num_mels, 1], 
                                                     dtype=np.float32), data[2][:,:,1:]), axis=2)
            except:
                raise TypeError('not same dimension')
            
            # 값들 Variable클래스에 저장
            if use_cuda:                
                characters = Variable(torch.from_numpy(data[0]).type(torch.cuda.LongTensor), requires_grad=False).cuda()
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.cuda.FloatTensor), requires_grad =False).cuda()
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.cuda.FloatTensor), requires_grad=False).cuda()

            else:
                characters = Variable(torch.from_numpy(data[0]).type(torch.cuda.LongTensor), requires_grad=False)
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor), requires_grad=False)
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.cuda.FloatTensor), requires_grad =False)
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.cuda.FloatTensor), requires_grad=False)

            # forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # loss caculate
            mel_loss = criterion(mel_output, mel_spectrogram) # mel_spectrogram(real value), mel_output(predicted value)
            linear_loss = torch.abs(linear_output - linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:]) # 해당 계산을 하는 이유??
            loss = mel_loss + linear_loss
            loss = loss.cuda()

            start_time = time.time() # training start time

            loss.backward() # 역전파

            #loss clip
            nn.utils.clip_grad_norm(mmodel.parameters(), 1.)

            optimizer.step() # gradient update

            time_per_step = time.time() - start_time # time per one step

            if current_step & hp.log_step == 0: # 100 step마다 한번 출력
                print('time per step : %.2f sec'% time_per_step)
                print("At timestep %d"% current_step)
                print('linear loss : %.4f'% linear_loss.data[0])
                print('mel loss: %.4f' % mel_loss.data[0])
                print('total loss : %.4f'% loss.data[0])

            if current_step % hp.save_step == 0: # 2000 step마다 기울기 저장
                save_checkpoint({'model' : model.state_dict()},
                                os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print('save model at step %d ...'% current_step)
                
            if current_step in hp.decay_step:
                optimizer  = adjust_learning_rate(optimizer, current_step)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    ''' 
    일정한 step이 지날 대마다 learning rate 변경
    optimizer_param_groups (dict) : parameter 정보들이 dict형태로 저장되어 있다.
    '''
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003
    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    args = parser.parse_args()
    main(args)

