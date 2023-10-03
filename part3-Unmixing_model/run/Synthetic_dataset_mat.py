import numpy as np
import scipy.io
import os
import pandas as pd
from mixed_fundamental_vectors import *


def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the ' + dirName + ' is created!---')
    else:
        print('---The dir is there!---')


def get_rate(m_rate):
    m_rate = m_rate.split('_')
    A_num = float(m_rate[0])
    B_num = float(m_rate[1])
    A_rate = A_num / (A_num + B_num)
    B_rate = B_num / (A_num + B_num)
    return A_rate, B_rate


def get_csvinfo2(A_num, B_num, csv_path, prob_path):
    csvinfo = pd.read_csv(csv_path)
    prob_np = np.load(prob_path)
    img_ids = csvinfo['Id'].values
    img_labels = csvinfo['Target'].values
    img_rates = csvinfo['m_rate'].values

    A_inds, B_inds, mix_inds = index_separate(img_labels, A_num, B_num)

    f_A = get_fundamental_pattern(prob_np, A_inds)
    f_B = get_fundamental_pattern(prob_np, B_inds)
    all_ids = mix_inds + A_inds + B_inds
    realLysorate = np.zeros(shape=(len(all_ids), 1), dtype=np.float)
    realMitorate = np.zeros(shape=(len(all_ids), 1), dtype=np.float)
    mixprobarr = np.zeros(shape=(len(all_ids), prob_np.shape[1]))
    i = 0

    for mix_id in all_ids:
        # get mixed rate
        mix_rate = img_rates[mix_id]
        # print(mix_rate)
        # if mix_rate == '0_0':
        #     continue
        Lyso_rate, Mito_rate = get_rate(mix_rate)
        realLysorate[i] = Lyso_rate
        realMitorate[i] = Mito_rate

        # get mix prob array
        mixprobarr[i, :] = prob_np[mix_id, :]
        i = i + 1

    return realLysorate.T, realMitorate.T, np.array([f_A]), np.array([f_B]), mixprobarr


# comb1: A_num=0, B_num=1  (CY,NP)
# comb2: A_num=0, B_num=4 (CY,PM)
# comb3: A_num=1, B_num=2   (NP,MT)
# comb4: A_num=1, B_num=3  (NP,NC)

SYN_LIST = {'img_comb1':[0,1],
            'img_comb2':[0,4],
            'img_comb3':[1,2],
            'img_comb4':[1,3],}


if __name__ == '__main__':
    DATA_DIR = '../HPA_imgs/Dataset'
    RESULT_DIR = '../result/submissions'
    SAVE_DIR = '../result/CNN_output'

    for syn_comb in SYN_LIST.keys():
        A_num = SYN_LIST[syn_comb][0]
        B_num = SYN_LIST[syn_comb][1]

        csv_path = os.path.join(DATA_DIR, syn_comb + '.csv')
        npy_dir = os.path.join(RESULT_DIR, syn_comb)

        for params in ['feat', 'prob']:
            # save_sub_dir = ''
            # npy_file = ''
            if params == 'feat':
                save_sub_dir = 'CNN_penult_feat'
                npy_file = 'prob_test.npy'
            else:
                save_sub_dir = 'CNN_last_feat'
                npy_file = 'prob_test.npy'

            save_dir = os.path.join(SAVE_DIR, save_sub_dir)
            mat_path = os.path.join(save_dir, syn_comb + '.mat')
            mkdir(save_dir)

            # get the size of Avgarr
            aug_list = os.listdir(npy_dir)
            exnpy = np.load(os.path.join(npy_dir, aug_list[0], npy_file))
            summix = np.zeros(exnpy.shape)
            sumbase = np.zeros((2, exnpy.shape[1]))
            Rate = np.zeros((exnpy.shape[0], 0))

            # test model with a aug_dir
            for aug_dir in aug_list:
                # 'prob_test.npy' for last features And 'features_test.npy' for penultimate features
                prob_path = os.path.join(npy_dir, aug_dir, npy_file)
                realLysorate, realMitorate, f_A, f_B, mixprobarr = get_csvinfo2(A_num, B_num, csv_path, prob_path)
                base = np.vstack((f_A, f_B))
                mix = mixprobarr
                rate = np.vstack((realLysorate, realMitorate))
                # print(base.shape)
                # print(mix.shape)
                # print(rate.shape)

                summix = summix + mix
                sumbase = sumbase + base
                Rate = rate
                # mat_path = os.path.join(save_dir, aug_dir+'.mat')
                # print(mat_path)
                # scipy.io.savemat(mat_path, {'base': base, 'mix': mix, 'rate': rate})
            MixMat = summix / len(aug_list)
            BaseMat = sumbase / len(aug_list)
            RateMat = Rate
            scipy.io.savemat(mat_path, {'MixMat': MixMat, 'BaseMat': BaseMat, 'RateMat': RateMat})

