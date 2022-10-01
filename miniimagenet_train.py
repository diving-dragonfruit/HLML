import argparse

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

from MiniImagenet import MiniImagenet
from meta import Meta
import json
from typing import List


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main(args, data_dict):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # changed architecture
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    print("device: ", device)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    print('Total trainable tensors:', num)

    root = 'miniimagenet/'
    # batchsz here means total episode number
    mini = MiniImagenet(root, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000*args.task_num, resize=args.imgsz)  # I multiplied batchsz by task_num to meet the number of epochs
    mini_test = MiniImagenet(root, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    # each epoch will run 10000 steps, each step runs 'args.task_num' tasks
    for epoch in range(args.epoch // 10000):
        print("Epoch: ", epoch)
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        print("DataLoader length: ", len(db))

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db, start=1):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            train_accs, train_loss = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 100 == 0:
                print('step:', step + 10000 * epoch, '\ttraining acc:', train_accs, '  training loss: {0:.4f}'.format(train_loss))

            if step % 1000 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                loss_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    test_acc, test_loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(test_acc)
                    loss_all_test.append(test_loss)

                # [b, update_step+1]
                avg_test_acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                avg_test_loss = np.mean(loss_all_test)
                print('Test acc:', avg_test_acc, '  Test loss: {0:.4f}'.format(avg_test_loss))
                
                record = len(data_dict['train_loss (per 1000 epochs)']) + 1
                if record % 10 == 0:
                    data_dict['train_loss (per 1000 epochs)'].append([record, train_loss])
                    data_dict['test_loss (per 1000 epochs)'].append([record, avg_test_loss])
                    data_dict['train_acc (per 1000 epochs)'].append([record, train_accs[-1].item()])
                    data_dict['test_acc (per 1000 epochs)'].append([record, avg_test_acc[-1].item()])
                else:
                    data_dict['train_loss (per 1000 epochs)'].append(train_loss)
                    data_dict['test_loss (per 1000 epochs)'].append(avg_test_loss)
                    data_dict['train_acc (per 1000 epochs)'].append(train_accs[-1].item())
                    data_dict['test_acc (per 1000 epochs)'].append(avg_test_acc[-1].item())
                
                # Serializing json
                json_object = json.dumps(data_dict, indent=4)
                
                # Writing to json
                with open(f"experiment_results/miniimagenet/k_val={args.k_qry}/{args.n_way}_way_{args.k_spt}_shot/{file_name}", "w") as outfile:
                    outfile.write(json_object)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run', type=int, help='which run', default=1)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--p_task_1', type=float, help='percentage lists for task loop', default=1)
    argparser.add_argument('--p_task_2', type=float, help='percentage lists for task loop', default=1)
    argparser.add_argument('--p_task_3', type=float, help='percentage lists for task loop', default=1)
    argparser.add_argument('--p_task_4', type=float, help='percentage lists for task loop', default=1)
    argparser.add_argument('--p_meta_1', type=float, help='percentage lists for meta loop', default=1)
    argparser.add_argument('--p_meta_2', type=float, help='percentage lists for meta loop', default=1)
    argparser.add_argument('--p_meta_3', type=float, help='percentage lists for meta loop', default=1)
    argparser.add_argument('--p_meta_4', type=float, help='percentage lists for meta loop', default=1)

    args = argparser.parse_args()
    args.p_task = [args.p_task_1, args.p_task_2, args.p_task_3, args.p_task_4]
    args.p_meta = [args.p_meta_1, args.p_meta_2, args.p_meta_3, args.p_meta_4]

    file_name = f'p_task={args.p_task};p_meta={args.p_meta};{args.run}th_run.json'
    print("run: ", args.run)
    
    data_dict = {
        'dataset': 'miniimagenet',
        'run': args.run,
        'hyperparameters': {
            'epoch': args.epoch, 'n_way': args.n_way, 'k_spt': args.k_spt, 'k_qry': args.k_qry, 
            'imgsz': args.imgsz, 'task_num': args.task_num, 
            'meta_lr': args.meta_lr, 'update_lr': args.update_lr,
            'update_step': args.update_step, 'update_step_test': args.update_step_test
        },
        'p_task': args.p_task,
        'p_meta': args.p_meta,
        'train_loss (per 1000 epochs)': [],
        'test_loss (per 1000 epochs)': [],
        'train_acc (per 1000 epochs)': [],
        'test_acc (per 1000 epochs)': [] 
    }
    
    main(args, data_dict=data_dict)
    
    # Serializing json
    json_object = json.dumps(data_dict, indent=4)
    
    # Writing to json
    with open(f"experiment_results/miniimagenet/k_val={args.k_qry}/{args.n_way}_way_{args.k_spt}_shot/{file_name}", "w") as outfile:
        outfile.write(json_object)

    print('miniimagenet, k_val = ', args.k_qry)
    print(f'{args.n_way}_way_{args.k_spt}_shot')
    print(file_name)