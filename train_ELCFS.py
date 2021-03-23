import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import collections
from collections import OrderedDict
from glob import glob

import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pytorch_metric_learning import losses

from networks.unet2d import Unet2D
from utils.losses import dice_loss
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from dataloaders.prepare_dataset import Dataset, ToTensor


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--client_num', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--display_freq', type=int, default=5, help='batch_size per gpu')

parser.add_argument('--unseen_site', type=int, default=0, help='batch_size per gpu')
args = parser.parse_args()

snapshot_path = "../output/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
meta_step_size = args.meta_step_size
clip_value = args.clip_value
base_lr = args.base_lr
client_num = args.client_num
max_epoch = args.max_epoch
display_freq = args.display_freq

client_name = ['Site1', 'Site2', 'Site3', 'Site4']
client_data_list = []
for client_idx in range(client_num):
    client_data_list.append(glob('/research/pheng4/qdliu/dataset/Fundus/{}/processed/npy/*'.format(client_name[client_idx])))
    print (len(client_data_list[client_idx]))
slice_num = np.array([101, 159, 400, 400])
volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
source_site_idx = [0, 1, 2, 3]
source_site_idx.remove(unseen_site_idx)
client_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight = np.round(client_weight, decimals=2)
client_weight[-1] = 1 - np.sum(client_weight[:2])
client_weight = np.insert(client_weight, unseen_site_idx, 0)
print(client_weight)
num_classes = 3

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def update_global_model(net_clients, client_weight):
    # Use the true average until the exponential average is more correct
    for param in zip(net_clients[0].parameters(), net_clients[1].parameters(), net_clients[2].parameters(), \
        net_clients[3].parameters()):

        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda() 
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)

def extract_contour_embedding(contour_list, embeddings):

    average_embeddings_list = []
    for contour in contour_list:
        contour_embeddings = contour * embeddings
        average_embeddings = torch.sum(contour_embeddings, (-1,-2))/torch.sum(contour, (-1,-2))
    # print (contour.shape)
    # print (embeddings.shape)
    # print (contour_embeddings.shape)
    # print (average_embeddings.shape)
        average_embeddings_list.append(average_embeddings)
    return average_embeddings_list

def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    haus_array = []

    for fid, filename in enumerate(test_data_list):
        data = np.load(filename)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float()

        logit, pred, _ = test_net(image)
        pred_y = pred.cpu().detach().numpy()
        pred_y[pred_y>0.75] = 1
        pred_y[pred_y<0.75] = 0

        pred_y_0 = pred_y[:, 0:1, ...]
        pred_y_1 = pred_y[:, 1:, ...]
        processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
        processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
        processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
        dice_subject = _eval_dice(mask, processed_pred_y)
        # haus_subject = _eval_haus(mask, processed_pred_y)
        dice_array.append(dice_subject)
        # haus_array.append(haus_subject)
    dice_array = np.array(dice_array)
    # print (dice_array.shape)
    dice_avg = np.mean(dice_array, axis=0).tolist()
    # print (dice_avg)
    # haus_avg = np.mean(haus_array, axis=0).tolist()[0]
    logging.info("OD dice_avg %.4f OC dice_avg %.4f" % (dice_avg[0], dice_avg[1]))
    return dice_avg, dice_array, 0, [0,0]

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    for client_idx in range(client_num):
        freq_site_idx = source_site_idx.copy()
        if client_idx != unseen_site_idx:
            freq_site_idx.remove(client_idx)
        dataset = Dataset(client_idx=client_idx, freq_site_idx=freq_site_idx,
                                split='train', transform = transforms.Compose([
                                ToTensor(),
                                ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
        net = Unet2D()
        net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        dataloader_clients.append(dataloader)
        net_clients.append(net)
        optimizer_clients.append(optimizer)

    for name, param in  net_clients[0].named_parameters():
        print (name)

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    # start federated learning
    writer = SummaryWriter(snapshot_path+'/log')
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for client_idx in source_site_idx:
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()
            iter_num = 0

            for i_batch, sampled_batch in enumerate(dataloader_current):
                time2 = time.time()

                # obtain training data
                volume_batch, label_batch, disc_contour, disc_bg, cup_contour, cup_bg = sampled_batch['image'], sampled_batch['label'], \
                sampled_batch['disc_contour'], sampled_batch['disc_bg'], sampled_batch['cup_contour'], sampled_batch['cup_bg']
                # volume_batch_raw = volume_batch[:, :3, ...]
                # volume_batch_trs_1 = volume_batch[:, 3:6, ...]
                # volume_batch_trs_2 = volume_batch[:, 6:, ...]
                # volume_batch_raw, volume_batch_trs_1, volume_batch_trs_2, label_batch = \
                #     volume_batch_raw.cuda(), volume_batch_trs_1.cuda(), volume_batch_trs_2.cuda(), label_batch.cuda()
                volume_batch_raw_np = volume_batch[:, :3, ...]
                volume_batch_trs_1_np = volume_batch[:, 3:6, ...]
                volume_batch_raw, volume_batch_trs_1, label_batch = \
                    volume_batch_raw_np.cuda(), volume_batch_trs_1_np.cuda(), label_batch.cuda()
                disc_contour, disc_bg, cup_contour, cup_bg = disc_contour.cuda(), disc_bg.cuda(), cup_contour.cuda(), cup_bg.cuda()

                # obtain updated parameter at inner loop
                outputs_soft_inner, outputs_mask_inner, embedding_inner = net_current(volume_batch_raw)
                loss_inner = dice_loss(outputs_soft_inner, label_batch)
                grads = torch.autograd.grad(loss_inner, net_current.parameters(), retain_graph=True)

                fast_weights = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, 0-clip_value, clip_value))) for
                                                  ((name, param), grad) in
                                                  zip(net_current.named_parameters(), grads))

                # outer loop evaluation
                outputs_soft_outer_1, outputs_mask_outer_1, embedding_outer = net_current(volume_batch_trs_1, fast_weights) #alpha
                loss_outer_1_dice = dice_loss(outputs_soft_outer_1, label_batch)

                inner_disc_ct_em, inner_disc_bg_em, inner_cup_ct_em, inner_cup_bg_em = \
                    extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_inner)
                outer_disc_ct_em, outer_disc_bg_em, outer_cup_ct_em, outer_cup_bg_em = \
                    extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_outer)

                disc_ct_em = torch.cat((inner_disc_ct_em, outer_disc_ct_em), 0)
                disc_bg_em = torch.cat((inner_disc_bg_em, outer_disc_bg_em), 0)
                cup_ct_em = torch.cat((inner_cup_ct_em, outer_cup_ct_em), 0)
                cup_bg_em = torch.cat((inner_cup_bg_em, outer_cup_bg_em), 0)
                disc_em = torch.cat((disc_ct_em, disc_bg_em), 0)
                cup_em = torch.cat((cup_ct_em, cup_bg_em), 0)
                label = np.concatenate([np.ones(disc_ct_em.shape[0]), np.zeros(disc_bg_em.shape[0])])
                label = torch.from_numpy(label)

                disc_cont_loss = cont_loss_func(disc_em, label)
                cup_cont_loss = cont_loss_func(cup_em, label)
                cont_loss = disc_cont_loss + cup_cont_loss
                loss_outer = loss_outer_1_dice + cont_loss * 0.1

                total_loss = loss_inner + loss_outer 

                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()

                iter_num = iter_num + 1
                if iter_num % display_freq == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/inner', loss_inner, iter_num)
                    writer.add_scalar('loss/outer', loss_outer, iter_num)
                    writer.add_scalar('loss/total', total_loss, iter_num)
                    logging.info('Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' % \
                        (epoch_num, client_idx, iter_num, len(dataloader_current), loss_inner.item(), loss_outer_1_dice.item(), cont_loss.item(), loss_outer.item(), total_loss.item()))

                if iter_num % 20 == 0:
                    image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                    writer.add_image('train/RawImage', image, iter_num)

                    image = np.array(volume_batch_trs_1_np[0, 0:3, :, :], dtype='uint8')
                    writer.add_image('train/TrsImage', image, iter_num)

                    image = outputs_soft_inner[0, 0:1, ...].data.cpu().numpy()
                    writer.add_image('train/RawDiskMask', image, iter_num)
                    image = outputs_soft_inner[0, 1:, ...].data.cpu().numpy()
                    writer.add_image('train/RawCupMask', image, iter_num)


                    image = np.array(disc_contour[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/disc_contour', image, iter_num)

                    image = np.array(disc_bg[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/disc_bg', image, iter_num)

                    image = np.array(cup_contour[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/cup_contour', image, iter_num)

                    image = np.array(cup_bg[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/cup_bg', image, iter_num)


        ## model aggregation
        update_global_model(net_clients, client_weight)

        ## evaluation
        with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
            dice_list = []
            haus_list = []
            print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
            dice, dice_array, haus, haus_array = test(unseen_site_idx, net_clients[unseen_site_idx])
            print(("   OD dice is: {}, std is {}".format(dice[0], np.std(dice_array[:, 0]))), file=f)
            print(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))), file=f)
            
        ## save model
        save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(epoch_num) + '.pth')
        torch.save(net_clients[0].state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    writer.close()

