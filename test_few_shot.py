import argparse
import subprocess

import yaml

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# Lemon:计算执行时间的装饰器。
def cul_time(f):
    def inner(*arg, **kwargs):
        start_time = time.time()
        re = f(*arg, **kwargs)
        print("程序执行时间为：", time.time()-start_time, "秒")
        return re
    return inner


# @cul_time

def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
        dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(
        dataset.label, n_batch, n_way, n_shot + n_query,
        ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=0, pin_memory=True)

    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    # return
    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}
    # aves= {'vl': <utils.Averager object at 0x7f64ecdb41d0>,
    # 'va': <utils.Averager object at 0x7f64ecdb4198>}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    n_query_per_epoch = n_batch * ep_per_batch * n_query * n_way

    test = False
    for epoch in range(1, test_epochs + 1):
        n_correct_classification = 0
        for data, _ in tqdm(loader, leave=False): # data.shape = torch.Size([25, 3, 80, 80])
            x_shot, x_query = fs.split_shot_query(
                data.cuda(), n_way, n_shot, n_query,
                ep_per_batch=ep_per_batch)
            # print(x_shot.shape) # torch.Size([1, 5, 1, 3, 80, 80])
            # print(x_query.shape) # torch.Size([1, 20, 3, 80, 80])

            with torch.no_grad():
                if not args.sauc:
                    # 这里开始分化，第一种是我的度量方式，第二种是作者的度量方式
                    logits = model(x_shot, x_query)  # 3,20,5
                    # print('hhh', logits.shape, type(logits))
                    if test:
                        n_correct_classification += logits
                        # print('logits={}, n_acc1={}'.format(logits, n_correct_classification))
                    else:
                        logits = logits.view(-1, n_way)  # torch.Size([60, 5])
                        label = fs.make_nk_label(n_way, n_query,
                                                 ep_per_batch=ep_per_batch).cuda()  # torch.Size([20])

                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)
                        # # acc = lemon.compute_acc(logits, label)

                        aves['vl'].add(loss.item(), len(data))
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                        *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(
                        bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)

        if test:
            print('test epoch {}: n_acc1={}, n_acc2={},acc={}'.format(
                epoch, n_correct_classification, n_query_per_epoch, n_correct_classification / n_query_per_epoch))
        else:
            print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                       mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), _[-1]))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)

    main(config)

    print("程序执行时间为：", time.time() - start_time, "秒")
