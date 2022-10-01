import torch, os
import numpy as np
from omniglotNShot import OmniglotNShot
import argparse

from meta import Meta

import json

def main(args, data_dict):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # this is the final architecture !!!
    config = [
        ('conv2d', [64, 1, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda')
    print("device: ", device)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                             batchsz=args.task_num,
                             n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry,
                             imgsz=args.imgsz)

    for step in range(1, args.epoch+1):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias.
        train_accs, train_loss = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 100 == 0:
            print('step:', step, '\ttraining acc:', train_accs, '  training loss: {0:.4f}'.format(train_loss))

        if step % 1000 == 0:
            test_accs = []
            test_losses = []

            for _ in range(1000 // args.task_num):  # I multiplied it by 2 or 1.5
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time.
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, test_loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    test_accs.append(test_acc)  # list of lists
                    test_losses.append(test_loss)  # list of scalars

            # [b, update_step+1]
            avg_test_acc = np.array(test_accs).mean(axis=0).astype(np.float16)  # a list
            avg_test_loss = np.mean(test_losses)  # a scalar
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
            with open(f"experiment_results/omniglot/k_val={args.k_qry}/{args.n_way}_way_{args.k_spt}_shot/{file_name}", "w") as outfile:
                outfile.write(json_object)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run', type=int, help='which run', default=1)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
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
        'dataset': 'omniglot',
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
    with open(f"experiment_results/omniglot/k_val={args.k_qry}/{args.n_way}_way_{args.k_spt}_shot/{file_name}", "w") as outfile:
        outfile.write(json_object)

    print('omniglot, k_val = ', args.k_qry)
    print(f'{args.n_way}_way_{args.k_spt}_shot')
    print(file_name)