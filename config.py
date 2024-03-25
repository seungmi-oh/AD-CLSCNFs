from __future__ import print_function
import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='the random seed number (default: the integer part of timestamp)')
    parser.add_argument('-run', '--run-name', default=0, type=int, 
                        help='name of the run (default: 0)')
    parser.add_argument("--gpu", default='0', type=str, 
                        help='GPU device number')
    parser.add_argument('--no_cuda', type=str2bool, default='no', 
                        help='disables CUDA training (default: no)')
    parser.add_argument('--workers', default=4, type=int, 
                        help='number of data loading workers (default: 4)')


    parser.add_argument('--dataset', default='mvtec', type=str, 
                        help='dataset name: mvtec (default: mvtec, btad)')
    parser.add_argument('--data_path', default='./../datasets/plain', type=str, 
                        help='path of dataset (default: ./datasets/plain)')
    parser.add_argument('--data_aug_path', default='./../datasets/aug_set', type=str, 
                        help='path of synthetic defect dataset for validation (default: ./datasets/aug_set)')
    parser.add_argument('--aug_sample_path', default='./../datasets/aug_samples', type=str, 
                        help='path to save synthetic defect samples (default: ./datasets/aug_samples)')
    parser.add_argument('-cl', '--class_name', default='none', type=str, 
                        help='class name for MVTec or BTAD (default: none)')
    parser.add_argument('-inp', '--input_size', default=256, type=int, 
                        help='image resize dimensions (default: 256)')
    parser.add_argument('--norm_mean', type=float, default=[0.485, 0.456, 0.406], nargs='+')
    parser.add_argument('--norm_std', type=float, default=[0.229, 0.224, 0.225], nargs='+')

    parser.add_argument('--aug_ratio_train', type=float, default=1.0, 
                        help='the ratio of synthetic defect data for training (default: 1.0)')
    parser.add_argument('--use_in_domain_data', type=str2bool, default='yes', 
                        help='use the in-domain dataset for generating synthetic defect data with dtd data together (50% selection, default: yes)')
    parser.add_argument('--repeat_num', type=int, default=5, 
                        help='period to generate new synthetic defect dataset (default: 5)')

    parser.add_argument('--model_path', default='./../models', type=str, 
                        help='root directory to save or load models (default: ./../models)')
    parser.add_argument('--drop_last', type=str2bool, default='no', 
                        help='do not train the last batch if the size of last minibatch is smaller than batch size (default: no)')

    parser.add_argument('-enc', '--enc_arch', default='wide_resnet50_2', type=str, 
                        help='feature extractor: wide_resnet50_2/resnet18 (default: wide_resnet50_2)')
    parser.add_argument('--pretrained', type=str2bool, default='yes', 
                        help='initialize the weight of feature extractor by that of pretrained network on ImageNet (default: yes)')

    parser.add_argument('-nf', '--nf_arch', default='freia-cflow', type=str, 
                        help='normalizing flow model (default: freia-cflow)')
    parser.add_argument('-pl', '--pool-layers', default=[1,2,3], type=int, 
                        help='number of layers used in CNF model (default: [1,2,3])', nargs='+')
    parser.add_argument('-cb', '--coupling-blocks', default=6, type=int, 
                        help='number of layers used in a CNF network (default: 6)')
    parser.add_argument('--clamp_alpha', type=float, default=1.9, 
                        help='hyper parameter of CNF models (default: 1.9)')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='hyper parameter of CNF models (default: 0)')
    parser.add_argument('--condition_vec', type=int, default=128, 
                        help='the dimenstion of positional embeddings (defect:128)')
    parser.add_argument('--p_bs', type=int, default=256, 
                        help='minibatch size of feature vectors for training CNF networks (default: 256)')

    parser.add_argument('--train_type', type=str, default='fe_only', choices = ['fe_only', 'nf_only', 'base-nf_only'], 
                        help='set which networks do you want to train. (fe_only: fine-tuning the feature extractor, nf_only: train CNF networks with the fine-tuned encoder, base-nf_only: train CNF networks with the pre-trianed nework, default: fe_only)')
    parser.add_argument('--infer_type', type=str, default='fe_only', choices = ['fe_only', 'nf_only', 'joint', 'base-nf_only', 'joint_not_sharing'],                        help='set which networks do you want to test. (fe_only: inference the enc-dec network, nf_only: inference CNF networks with the fine-tuned encoder, joint: inference tne enc-dec network and CNF networks with the shared feature extractor, base-nf_only: inference CNF networks with the pre-trianed nework, joint_not_sharing: inference tne enc-dec network and CNF networks without the shared feature extractor, default: fe_only)')
    parser.add_argument('--test_data_type', type=str, default='real', choices = ['aug', 'real'], 
                        help = 'set which dataset do you want to evaluate. (aug: synthetic defect dataset for evaluation, real: real test datastet provided from MVTecAD dataset, default: real)')
    parser.add_argument('--loss_type', type=str, default='smooth_cls', choices = ['cls', 'reg', 'smooth_cls'], 
                        help='set which task do you want to fine-tune the encoder. (cls: pixelwise classification with hard labels, reg: pixelwise regression network, smooth_cls: pixelwise classification network with soft labels, default: smooth_cls)')
    parser.add_argument('--get_best_w_fe', type=str2bool, default='no', 
                        help='find the best weight to balance the anomaly scores of the pixelwise classification and CNF networks. (default: no)')
    parser.add_argument('--skip_connection', type=str2bool, default='yes', 
                        help='determine whether using skip connections to construct a pixelwise classification(or regression) network. (default: no)')
    parser.add_argument('--set_dec_dims_nf', type=str2bool, default='yes')
    parser.add_argument('--num_class', type=int, default=2, 
                        help='the number of class for a pixelwise classification network. (default: 2)')

    parser.add_argument('-bs', '--batch-size', default=32, type=int,
                        help='train batch size (default: 32)')
    parser.add_argument('--meta_epochs', type=int, default=25,
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=8,
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('--freeze_enc_epochs', type=int, default=5)
    parser.add_argument("--is_train", default='yes', type=str2bool,
                        help=' yes-train/no-inference (default: yes)')

    parser.add_argument('--pro',  type = str2bool, default='no',
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz',  type = str2bool, default='no',
                        help='saves test data visualizations')

    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs_percentage', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], nargs='+', 
                        help='epochs to decay learning rate for the StepLR decay scheduling.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help = 'multiplicative factor of learning rate decay. (default: 0.1)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, 
                        help='epochs to increase learning rate up to the initial learning rate in early training steps. (default: 2)')
    parser.add_argument('--lr_warm', type=str2bool, default='yes', 
                        help= 'determine whether using warmup learning rate scheduling. (default: yes)')
    parser.add_argument('--lr_cosine', type=str2bool, default='yes', 
                        help='yes-CosineAnnealingLR scheduling, no-StepLR scheduling. (default: yes)')

    parser.add_argument('--w_fe', type=float, default=0, 
                        help='the weight to balance the anomaly scores of the pixelwise classification and CNF networks.')
    parser.add_argument('--w_defect', type=float, default=0.9,
                        help='the weight of defect class for weighted cross-entropy. (default:0.9)')
    parser.add_argument('--w_decay', type=float, default=1e-4,
                        help='the hyperparameter for the weight decay regularization. (default:1e-4)')

    parser.add_argument('--feat_avg_topk', type=float, default=1.0, 
                        help='select feature maps with the k larges averaged values and average them to visualize the features.')
    parser.add_argument('--add_fe_anomaly', type = str2bool, default='no',
                        help='determine whether combining the score maps of a pixelwise classification network and CNF networks.')
    parser.add_argument('--th_manual', type = float, default=0,
                        help='manual threshold value to make predictions from anomaly scores. (default: 0)')
    parser.add_argument('--is_close', type = str2bool, default='no',
                        help='determine whether applying the morphological closing operation to the prediction map. (default: no)')
    parser.add_argument('--is_open', type = str2bool, default='yes',
                        help='determine whether applying the morphological opening operation to the prediction map. (default: yes)')
    parser.add_argument('--is_k_disk', type = str2bool, default='yes',
                        help='determine whether using the disk type mask for the morphological operations. (default: yes)')
    parser.add_argument('--k_size', default=1, type=int, 
                        help='the hyperparameter for the size of the mask for the morphological operations. (default: 1)')

    # output settings
    parser.add_argument('--verbose', type=str2bool, default='yes')
    parser.add_argument('--hide_tqdm_bar', type=str2bool, default='yes')
    parser.add_argument('--save_results', type=str2bool, default='yes')

    args = parser.parse_args()
    
    return args

# convert input string to the boolean type 
def str2bool(v):
   if isinstance(v, bool):
       return v
   if v.lower() in ('yes', 'True', 't', 'y'):
       return True
   elif v.lower() in ('no', 'false', 'f', 'n'):
       return False
   else:
       raise argparse.ArgumentTypeError('Boolean value expected.')
