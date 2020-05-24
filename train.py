import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt

from torchvision import transforms

MEAN = 0.5
STD = 0.5

NUM_WORKER = 0


def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png', 'a2b'))
        os.makedirs(os.path.join(result_dir_train, 'png', 'b2a'))

    ## 네트워크 학습하기
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              Normalization(mean=MEAN, std=STD)])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train,
                                task=task, data_type='both')
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## 네트워크 생성하기
    if network == "CycleGAN":
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=9).to(device)
        netG_b2a = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=9).to(device)

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)
        netD_b = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)

        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG_a2b, netG_b2a, \
            netD_a, netD_b, \
            optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                                            netD_a=netD_a, netD_b=netD_b,
                                            optimG=optimG, optimD=optimD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_G_a2b_train = []
            loss_G_b2a_train = []
            loss_D_a_train = []
            loss_D_b_train = []
            loss_cycle_a_train = []
            loss_cycle_b_train = []
            loss_ident_a_train = []
            loss_ident_b_train = []

            for batch, data in enumerate(loader_train, 1):
                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimD.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())

                loss_D_a_real = fn_gan(pred_real_a, torch.ones_like(pred_real_a))
                loss_D_a_fake = fn_gan(pred_fake_a, torch.zeros_like(pred_fake_a))
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())

                loss_D_b_real = fn_gan(pred_real_b, torch.ones_like(pred_real_b))
                loss_D_b_fake = fn_gan(pred_fake_b, torch.zeros_like(pred_fake_b))
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                loss_G_a2b = fn_gan(pred_fake_a, torch.ones_like(pred_fake_a))
                loss_G_b2a = fn_gan(pred_fake_b, torch.ones_like(pred_fake_b))

                loss_cycle_a = fn_cycle(input_a, recon_a)
                loss_cycle_b = fn_cycle(input_b, recon_b)

                ident_a = netG_b2a(input_a)
                ident_b = netG_a2b(input_b)

                loss_ident_a = fn_ident(input_a, ident_a)
                loss_ident_b = fn_ident(input_b, ident_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         wgt_cycle * (loss_cycle_a + loss_cycle_b) + \
                         wgt_cycle * wgt_ident * (loss_ident_a + loss_ident_b)

                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]

                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]

                loss_cycle_a_train += [loss_cycle_a.item()]
                loss_cycle_b_train += [loss_cycle_b.item()]

                loss_ident_a_train += [loss_ident_a.item()]
                loss_ident_b_train += [loss_ident_b.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN a2b %.4f b2a %.4f | "
                      "DISC a %.4f b %.4f | "
                      "CYCLE a %.4f b %.4f | "
                      "IDENT a %.4f b %.4f | " %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_a2b_train), np.mean(loss_G_b2a_train),
                       np.mean(loss_D_a_train), np.mean(loss_D_b_train),
                       np.mean(loss_cycle_a_train), np.mean(loss_cycle_b_train),
                       np.mean(loss_ident_a_train), np.mean(loss_ident_b_train)))

                if batch % 20 == 0:
                    # Tensorboard 저장하기
                    input_a = fn_tonumpy(fn_denorm(input_a)).squeeze()
                    input_b = fn_tonumpy(fn_denorm(input_b)).squeeze()
                    output_a = fn_tonumpy(fn_denorm(output_a)).squeeze()
                    output_b = fn_tonumpy(fn_denorm(output_b)).squeeze()

                    input_a = np.clip(input_a, a_min=0, a_max=1)
                    input_b = np.clip(input_b, a_min=0, a_max=1)
                    output_a = np.clip(output_a, a_min=0, a_max=1)
                    output_b = np.clip(output_b, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', '%04d_input_a.png' % id), input_a[0],
                               cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', '%04d_output_b.png' % id), output_b[0],
                               cmap=cmap)

                    plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', '%04d_input_b.png' % id), input_b[0],
                               cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', '%04d_output_a.png' % id), output_a[0],
                               cmap=cmap)

                    writer_train.add_image('input_a', input_a, id, dataformats='NHWC')
                    writer_train.add_image('input_b', input_b, id, dataformats='NHWC')
                    writer_train.add_image('output_a', output_a, id, dataformats='NHWC')
                    writer_train.add_image('output_b', output_b, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_a2b', np.mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', np.mean(loss_G_b2a_train), epoch)

            writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)
            writer_train.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch)

            writer_train.add_scalar('loss_cycle_a', np.mean(loss_cycle_a_train), epoch)
            writer_train.add_scalar('loss_cycle_b', np.mean(loss_cycle_b_train), epoch)

            writer_train.add_scalar('loss_ident_a', np.mean(loss_ident_a_train), epoch)
            writer_train.add_scalar('loss_ident_b', np.mean(loss_ident_b_train), epoch)

            if epoch % 2 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, epoch=epoch,
                     netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                     netD_a=netD_a, netD_b=netD_b,
                     optimG=optimG, optimD=optimD)

        writer_train.close()


def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png', 'a2b'))
        os.makedirs(os.path.join(result_dir_test, 'png', 'b2a'))
        # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=MEAN, std=STD)])

        dataset_test_a = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task,
                                 data_type='a')
        loader_test_a = DataLoader(dataset_test_a, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_a = len(dataset_test_a)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

        dataset_test_b = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task,
                                 data_type='b')
        loader_test_b = DataLoader(dataset_test_b, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_b = len(dataset_test_b)
        num_batch_test_b = np.ceil(num_data_test_b / batch_size)

    ## 네트워크 생성하기
    if network == "CycleGAN":
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=9).to(device)
        netG_b2a = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=9).to(device)

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)
        netD_b = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)

        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG_a2b, netG_b2a, \
        netD_a, netD_b, \
        optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                        netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                                        netD_a=netD_a, netD_b=netD_b,
                                        optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()

            for batch, data in enumerate(loader_test_a, 1):
                # forward pass
                input_a = data['data_a'].to(device)

                output_b = netG_a2b(input_a)

                # Tensorboard 저장하기
                input_a = fn_tonumpy(fn_denorm(input_a))
                output_b = fn_tonumpy(fn_denorm(output_b))

                for j in range(input_a.shape[0]):
                    id = batch_size * (batch - 1) + j

                    input_a_ = input_a[j]
                    output_b_ = output_b[j]

                    input_a_ = np.clip(input_a_, a_min=0, a_max=1)
                    output_b_ = np.clip(output_b_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', '%04d_input_a.png' % id), input_a_)
                    plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', '%04d_output_b.png' % id), output_b_)

                    print("TEST A: BATCH %04d / %04d | " % (id + 1, num_data_test_a))

            for batch, data in enumerate(loader_test_b, 1):
                # forward pass
                input_b = data['data_b'].to(device)

                output_a = netG_b2a(input_b)

                # Tensorboard 저장하기
                input_b = fn_tonumpy(fn_denorm(input_b))
                output_a = fn_tonumpy(fn_denorm(output_a))

                for j in range(input_b.shape[0]):
                    id = batch_size * (batch - 1) + j

                    input_b_ = input_b[j]
                    output_a_ = output_a[j]

                    input_b_ = np.clip(input_b_, a_min=0, a_max=1)
                    output_a_ = np.clip(output_a_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', '%04d_input_b.png' % id), input_b_)
                    plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', '%04d_output_a.png' % id), output_a_)

                    print("TEST B: BATCH %04d / %04d | " % (id + 1, num_data_test_b))