import os
import argparse
import torch
from networks.unet2d import Unet2D
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
import numpy as np
from glob import glob
import logging
# from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='1030-meta-shape-site3', help='model_name')
parser.add_argument('--method', type=str,  default='epi_result', help='model_name')
parser.add_argument('--batch_size', type=int,  default=4, help='model_name')
parser.add_argument('--client_num', type=int,  default=4, help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--unseen_site', type=int,  default=3, help='GPU to use')
parser.add_argument('--model_idx', type=int,  default=85, help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
model_path = "../output/"+FLAGS.model+"/"
snapshot_path = "../output/"+FLAGS.method+"/"
# test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
# if not os.path.exists(test_save_path):
#     os.makedirs(test_save_path)
args = parser.parse_args()
batch_size = args.batch_size * len(args.gpu.split(','))
volume_size = [384, 384, 1]
num_classes = 2
client_num = args.client_num

client_name = ['Site1', 'Site2', 'Site3', 'Site4']

client_data_list = []
for client_idx in range(client_num):
    client_data_list.append(glob('/research/pheng4/qdliu/dataset/Fundus/{}/processed/npy/*'.format(client_name[client_idx])))
    print (len(client_data_list[client_idx]))

unseen_site_idx = args.unseen_site
source_site_idx = [0, 1, 2, 3]

result_dir = snapshot_path + '/prediction/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def _save_image(img, gth, pred, out_folder, out_name):

    np.save(out_folder+'/'+out_name+'_img.npy',img)
    np.save(out_folder+'/'+out_name+'_pred.npy',pred)
    np.save(out_folder+'/'+out_name+'_gth.npy',gth)

    return 0

def test(site_index, test_net_idx):
    
    test_net = Unet2D()
    test_net = test_net.cuda()

    save_mode_path = os.path.join(model_path + '/model', 'epoch_' + str(test_net_idx) + '.pth')
    test_net.load_state_dict(torch.load(save_mode_path))
    test_net.train()

    test_data_list = client_data_list[site_index]

    dice_array = []
    haus_array = []

    for fid, filename in enumerate(test_data_list):
        # if 'S-5-L' not in filename:
        #     continue
        print(filename)
        data = np.load(filename)
        image = data[..., :3]#np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = data[..., 3:]#np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(mask.transpose(2, 0, 1), axis=0)
        pred_y_list = []

        image_test = np.expand_dims(image.transpose(2, 0, 1), axis=0)

        image_test = torch.from_numpy(image_test).float()

        logit, pred, _ = test_net(image_test)
        pred_y = pred.cpu().detach().numpy()
        pred_y[pred_y>0.75] = 1
        pred_y[pred_y<0.75] = 0

        pred_y_0 = pred_y[:, 0:1, ...]
        pred_y_1 = pred_y[:, 1:, ...]
        processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
        processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
        processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
        dice_subject = _eval_dice(mask, processed_pred_y)
        haus_subject = _eval_haus(mask, processed_pred_y)
        dice_array.append(dice_subject)
        haus_array.append(haus_subject)

        _save_image(image.transpose(2, 0, 1), mask[0], pred_y[0], result_dir, out_name=str(site_index)+'_'+os.path.basename(filename))
    dice_array = np.array(dice_array)
    haus_array = np.array(haus_array)

    dice_avg = np.mean(dice_array, axis=0).tolist()
    haus_avg = np.mean(haus_array, axis=0).tolist()

    return dice_avg, dice_array, haus_avg, haus_array

if __name__ == '__main__':
    test_net_idx = args.model_idx

    with open(os.path.join(snapshot_path, 'testing_result.txt'), 'a') as f:
        # for test_net_idx in range(10,11):


            dice_list = []
            haus_list = []
            print("epoch {} testing ".format(test_net_idx))
            dice, dice_array, haus, haus_array = test(unseen_site_idx, test_net_idx)
            print(("   OD dice is: {}, std is {}, array is {}".format(dice[0], np.std(dice_array[:, 0]), dice_array[:, 0])), file=f)
            print(("      {}".format(dice_array[:, 0])), file=f)
            print(("   OC dice is: {}, std is {}, array is {}".format(dice[1], np.std(dice_array[:, 1]), dice_array[:, 1])), file=f)
            print(("      {}".format(dice_array[:, 1])), file=f)
            print ((dice[0]+dice[1])/2)
