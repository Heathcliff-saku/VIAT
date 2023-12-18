from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'nerf_for_attack'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest', 'test'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')


    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--batch_size_render', type=int, default=1,
                        help='images num for mutil gpus')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')                    
                        
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    parser.add_argument('--search_index', type=str,
                        default='th_phi',
                        choices=['th_phi', 'th', 'phi', 'r'],
                        help='search_index')
    parser.add_argument('--th_range', nargs="+", type=int, default=[-180, 180],
                        help='th_range')
    parser.add_argument('--phi_range', nargs="+", type=int, default=[-180, 0],
                        help='phi_range')
    parser.add_argument('--r_range', nargs="+", type=int, default=[4, 6],
                        help='r_range')
    parser.add_argument('--num_sample', type=int, default=100,
                        help='num_sample')

    parser.add_argument('--optim_method', type=str, default='random',
                        choices=['random', 'NES', 'xNES', 'train_pose'],
                        help='num_sample')

    parser.add_argument('--search_num', type=int, default=3,
                        help='search_num')

    parser.add_argument('--target_flag', type=bool, default=False,
                        help='target_flag')

    parser.add_argument('--target_label', type=int, default=584,
                        help='target_label')

    parser.add_argument('--popsize', type=int, default=21,
                        help='popsize')

    parser.add_argument('--iteration', type=int, default=20,
                        help='iteration')
    parser.add_argument('--iteration_warmstart', type=int, default=20,
                        help='iteration')

    

    parser.add_argument('--mu_lamba', type=float, default=0.0001,
                        help='iteration')
    parser.add_argument('--sigma_lamba', type=float, default=0.0001,
                        help='iteration')
    parser.add_argument('--omiga_lamba', type=float, default=0.0001,
                        help='iteration')

    parser.add_argument('--random_eplison', type=float, default=0.01,
                        help='iteration')

    parser.add_argument('--index', type=float, default=0,
                        help='iteration')

    parser.add_argument('--th_size', type=float, default=180,
                        help='iteration')
    parser.add_argument('--gamma_size', type=float, default=30,
                        help='iteration')
    parser.add_argument('--phi_size', type=float, default=70,
                        help='iteration')

    parser.add_argument('--r_size', type=float, default=1,
                        help='iteration')
    parser.add_argument('--x_size', type=float, default=0.5,
                        help='iteration')
    parser.add_argument('--y_size', type=float, default=0.5,
                        help='iteration')

    parser.add_argument('--num_pose', type=int, default=0,
                        help='iteration')

    parser.add_argument('--random_begin', default=False,
                        help='mu random init')
    
    parser.add_argument('--num_k', type=int, default=5,
                        help='mu random init')

    parser.add_argument('--n_jobs', type=int, default=5,
                        help='downsample factor (<=1.0) for the images')

    

    '测试环节'
    # 当前攻击的正确标签 名字！
    parser.add_argument('--label_name', type=str, default='hotdog, hot dog, red hot',help='当前攻击的正确标签，如果是有目标则为目标标签')
    # 当前攻击的目标数字标签，适应度评价用
    parser.add_argument('--label', type=int, default=934,help='当前攻击的正确标签，如果是有目标则为目标标签')


    # for AT:
    parser.add_argument('--AT_exp_name', type=str, default='background')
    parser.add_argument('--train_mood', type=str, default='train')
    
    parser.add_argument('--batch-size', type=int, default=72*4, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=72*4, metavar='N',
                    help='input batch size for testing (default: 128)')

    parser.add_argument('--crop_size', type=int, default=224, metavar='N',
                    help='crop_size')
    parser.add_argument('--img_size', type=int, default=256, metavar='N',
                        help='img_size')

    parser.add_argument('--epochs', type=int, default=76, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.031,
                        help='perturbation')
    parser.add_argument('--num-steps', default=1,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007,
                        help='perturb step size')
    parser.add_argument('--beta', default=6.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-dir', default='./model-imagenet-100-ckpts',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                        help='save frequency')

    parser.add_argument('--no_background', action='store_true', default=False)
    parser.add_argument('--share_dist', action='store_true', default=False)
    parser.add_argument('--fast_AVDT', action='store_true', default=False)
    parser.add_argument('--treat_model', type=str, default='resnet50')
    parser.add_argument('--ckpt_attack_path', default='./ckpts/nerf')
    parser.add_argument('--AT_type', type=str, default='AVDT', choices=['AVDT', 'natural', 'random'])
    parser.add_argument('--share_dist_rate', type=float, default=0.5, help='perturb step size')
    


    return parser.parse_args()
