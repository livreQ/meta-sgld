import  torch, os
import  numpy as np
from    src.utils.omniglotNShot import OmniglotNShot
import  argparse
from scipy.special import comb
from  src.algo.meta_sgld import Meta

def main(args):


    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    maml.env_task_num = comb(1200, maml.n_way)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,# fix task num at each T iter
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    for step in range(args.epoch): # one meta update
        #[b, nk, (feature size, eg. 28, 28) ]
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs, train_loss, norm_bound, inco_bound= maml(x_spt, y_spt, x_qry, y_qry)#meta param update once

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 200 == 0:
            accs = []
            test_loss = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )
                    test_loss.append(loss)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            print("Test\tmeta loss:", np.mean(test_loss))
            print("epoch:{},Train-Test loss gap:{:.4}, norm_bound:{:.4}, inco_bound:{:.4}".format(step, np.abs(train_loss - np.mean(test_loss)), norm_bound, inco_bound))



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--loss', type=str, help='specify loss function', default='cross_entropy')
    argparser.add_argument('--add_noise', type=int, help='add noise for each update', default=1)
    argparser.add_argument('--sample_stocha', type=int, help='random sample for subtasks', default=1)
    argparser.add_argument('--eva_bound', type=int, help='evaluate bound or not', default=1)
    argparser.add_argument('--mc_times', type=int, help='mc times', default=5)
    argparser.add_argument('--task', type=str, help='task type', default='classification')
    argparser.add_argument('--sample_num', type=int, help='sample batch size, namely sample num', default=2) 
    argparser.add_argument('--temp', type=int, help='sgld temparature', default=100000000) 
    argparser.add_argument('--inner_decay_step', type=int, help='inner decay', default=600) 
    argparser.add_argument('--outer_decay_step', type=int, help='outer decay', default=800) 
    argparser.add_argument('--env_task_num', type=int, help='all tasks avaible', default=20000)
    argparser.add_argument('--seed', type=int, help='random seed', default=222)
    args = argparser.parse_args()

    main(args)
