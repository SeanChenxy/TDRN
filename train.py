import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AnnotationTransform, BaseTransform, VOCDetection, detection_collate, coco_detection_collate, seq_detection_collate, mb_cfg, dataset_training_cfg, COCOroot, COCODetection
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import PriorBox
import numpy as np
import time
import logging
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def print_log(args):
    logging.info('model_name: '+ args.model_name)
    logging.info('ssd_dim: '+ str(args.ssd_dim))
    logging.info('Backbone: '+ args.backbone)
    logging.info('BN: '+ str(args.bn))
    logging.info('Conv7 Channel: '+ str(args.c7_channel))
    if 'RefineDet' in args.backbone:
        logging.info('Refine: ' + str(args.refine))
        logging.info('Deform: ' + str(args.deform))
        logging.info('Multi-head: ' + str(args.multihead))
    if args.resume:
        logging.info('resume: '+ args.resume )
        logging.info('start_iter: '+ str(args.start_iter))
    elif args.resume_from_ssd:
        logging.info('resume_from_ssd: '+ args.resume_from_ssd )
    else:
        logging.info('load pre-trained backbone: '+ args.basenet )
    logging.info('lr: '+ str(args.lr))
    logging.info('warm_epoch: '+ str(args.warm_epoch))
    logging.info('gamam: '+ str(args.gamma))
    logging.info('step_list: '+ str(args.step_list))
    logging.info('save_interval: '+ str(args.save_interval))
    logging.info('dataset_name: '+ args.dataset_name )
    logging.info('set_file_name: '+ args.set_file_name )
    logging.info('gpu_ids: '+ args.gpu_ids)
    logging.info('augm_type: '+ args.augm_type)
    logging.info('batch_size: '+ str(args.batch_size))
    logging.info('loss weights: '+ str(args.loss_coe))
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--basenet', default='vgg16bn_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint') #'./weights/tssd300_VID2017_b8s8_RSkipTBLstm_baseAugmDrop2Clip5_FixVggExtraPreLocConf/ssd300_seqVID2017_20000.pth'
parser.add_argument('--resume_from_ssd', default=None, type=str, help='Resume vgg and extras from ssd checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='./weights040/test', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='VOC0712', help='VOC0712/VIDDET/seqVID2017/MOT17Det/seqMOT17Det')
parser.add_argument('--step_list', nargs='+', type=int, default=[30,50], help='step_list for learning rate')
parser.add_argument('--backbone', default='RefineDet_ResNet101', type=str, help='Backbone')
parser.add_argument('--c7_channel', default=1024, type=int, help='out_channel of Conv7 in VGG')
parser.add_argument('--refine', default=True, type=str2bool, help='Only work when backbone==RefineDet')
parser.add_argument('--deform', default=1, type=int, help='number of deform group. 0: Do not use defomable conv. Only work when backbone==RefineDet')
parser.add_argument('--multihead', default=True, type=str2bool, help='Multihead detection')
parser.add_argument('--drop', default=1.0, type=float, help='DropOut, Only work when backbone==RefineDet')
parser.add_argument('--model_name', default='ssd', type=str, help='which model selected')
parser.add_argument('--ssd_dim', default=320, type=int, help='ssd_dim 300, 320 or 512')
parser.add_argument('--gpu_ids', default='4,5', type=str, help='gpu number')
parser.add_argument('--augm_type', default='base', type=str, help='how to transform data')
parser.add_argument('--set_file_name',  default='train', type=str, help='train_VID_DET/train_video_remove_no_object/train, MOT dataset does not use it')
parser.add_argument('--loss_coe', nargs='+', type=float, default=[1.0,1.0, 0.5], help='coefficients for loc, conf, att, asso')
parser.add_argument('--bn', default=False, type=str2bool, help='select sequence data in a skip way')
parser.add_argument('--save_interval', default=10, type=int, help='frequency of checkpoint saving')
parser.add_argument('--warm_epoch', default=0, type=int, help='warm epoch')
args = parser.parse_args()
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
current_time = time.strftime("%b_%d_%H:%M:%S_%Y", time.localtime())
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(args.save_folder, current_time+'.log'),
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
print_log(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
ssd_dim = args.ssd_dim
if args.dataset_name in ['MOT15', 'seqMOT15']:
    prior = 'MOT_300'
    cfg = mb_cfg[prior]
else:
    prior = 'VOC_'+ str(args.ssd_dim)
    if args.ssd_dim==300 and 'RFB' in args.backbone:
        prior += '_RFB'
    elif args.ssd_dim==512 and 'RefineDet' in args.backbone:
        prior += '_RefineDet'
    cfg = mb_cfg[prior]
train_sets, num_classes, data_root = dataset_training_cfg[args.dataset_name]
logging.info('train sets: ' + str(train_sets))
set_filename = args.set_file_name
if args.dataset_name[:3] == 'seq':
    collate_fn = seq_detection_collate
elif args.dataset_name == 'COCO':
    collate_fn = coco_detection_collate
else:
    collate_fn = detection_collate
if args.dataset_name == 'UW':
    means = (128, 128, 128)
else:
    means = (104, 117, 123)
mean_np = np.array(means, dtype=np.int32)
batch_size = args.batch_size
weight_decay = args.weight_decay
max_epoch = args.step_list[-1]
gamma = 0.1
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()

if 'RFB' in args.backbone:
    from model.rfbnet_vgg import build_net
    ssd_net = build_net('train', ssd_dim, num_classes, bn=args.bn)
elif 'RefineDet' in args.backbone:
    if 'MobNet' in args.backbone:
        if args.deform:
            from model.dualrefinedet_mobilenet import build_net
            ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes,
                            def_groups=args.deform, multihead=args.multihead)
        else:
            from model.refinedet_mobilenet import build_net
            ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, use_refine=args.refine)
    elif args.deform:
        from model.dualrefinedet_vggbn import build_net
        ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel, def_groups=args.deform, bn=args.bn, multihead=args.multihead)
    else:
        from model.refinedet_vgg import build_net
        ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, use_refine=args.refine, c7_channel=args.c7_channel, bn=args.bn, multihead=args.multihead)
elif 'MobNet' in args.backbone:
    from model.ssd4scale_mobile import build_net
    ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel)
elif '4s' in args.backbone:
    from model.ssd4scale_vgg import build_net
    ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel, bn=args.bn)
else:
    ssd_net = None
net = ssd_net
if device==torch.device('cuda'):
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
print(ssd_net)
net = net.to(device)

if args.resume:
    logging.info('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    backbone_weights = torch.load('/home/sean/Documents/SSD/weights040/'+ args.basenet)
    logging.info('Loading base network...')
    ssd_net.backbone.load_state_dict(backbone_weights)

if not args.resume:
    from model.networks import net_init
    net_init(ssd_net, args.backbone, logging, refine=args.refine, deform=args.deform, multihead=args.multihead)

if args.augm_type == 'ssd':
    data_transform = SSDAugmentation
else:
    data_transform = BaseTransform

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# criterion
if 'RefineDet' in args.backbone and args.refine:
    use_refine = True
    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, device=device, only_loc=True)
    criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, device=device)
else:
    use_refine = False
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, device=device)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward().to(device)

def train():
    net.train()
    epoch = args.start_iter
    if args.dataset_name == 'COCO':
        dataset = COCODetection(COCOroot, year='trainval2014', image_sets=train_sets, transform=data_transform(ssd_dim, means), phase='train')
    else:
        dataset = VOCDetection(data_root, train_sets, data_transform(ssd_dim, means),
                               AnnotationTransform(dataset_name=args.dataset_name),
                               dataset_name=args.dataset_name, set_file_name=set_filename)
    epoch_size = len(dataset) // args.batch_size
    drop_step = [s * epoch_size for s in args.step_list]
    max_iter = max_epoch * epoch_size
    logging.info('Loading Dataset:' + args.dataset_name + ' dataset size: ' +str(len(dataset)))

    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        y_dim = 3
        legend = ['Loss', 'Loc Loss', 'Conf Loss',]
        if use_refine:
            y_dim += 1
            legend += ['Arm Loc Loss',]

        lot = viz.line(
            X=torch.zeros((1,)),
            Y=torch.zeros((1, y_dim)),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title=args.save_folder.split('/')[-1],
                legend=legend,
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers, shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=True)

    for iteration in range(epoch*epoch_size, max_iter + 10):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            if epoch % args.save_interval == 0:
                logging.info('Saving state, epoch: '+ str(epoch))
                torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.model_name + str(
                    ssd_dim) + '_' + args.dataset_name + '_' +repr(epoch) + '.pth'))
            epoch += 1

        t0 = time.time()
        if iteration in drop_step:
            step_index = drop_step.index(iteration) + 1
        adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)
            # adjust_learning_rate(optimizer, args.gamma)

        collected_data = next(batch_iterator)
        with torch.no_grad():
            images, targets = collected_data[:2]
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

        # forward
        loss = torch.tensor(0., requires_grad=True).to(device)
        out = net(images)
        # backward
        optimizer.zero_grad()
        if use_refine:
            loss_arm_l = arm_criterion(out[0], priors, targets)
            loss_l, loss_c = criterion(out[2:], priors, targets, arm_data=out[:2])
            loss += args.loss_coe[0] * loss_arm_l

        else:
            loss_l, loss_c = criterion(out, priors, targets)
        loss += args.loss_coe[0] * loss_l + args.loss_coe[1] * loss_c

        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % 10 == 0:
            if use_refine:
                logging.info('Epoch:' + repr(epoch) + ', epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + ', total_iter ' + repr(
                    iteration) + ' || loss: %.4f, Loss_l: %.4f, loss_c: %.4f, loss_arm_l: %.4f, lr: %.5f || Timer: %.4f sec.' % (
                             loss, loss_l, loss_c,loss_arm_l, optimizer.param_groups[0]['lr'], t1 - t0))
            else:
                logging.info('Epoch:' + repr(epoch) + ', epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + ', total_iter ' + repr(
                    iteration) + ' || loss: %.4f, Loss_l: %.4f, loss_c: %.4f, lr: %.5f || Timer: %.4f sec.' % (loss, loss_l, loss_c, optimizer.param_groups[0]['lr'], t1 - t0))

        if args.visdom:
            y_dis = [loss.cpu(), args.loss_coe[0]*loss_l.cpu(), args.loss_coe[1]*loss_c.cpu()]
            if iteration == 1000:
                # initialize visdom loss plot
                lot = viz.line(
                    X=torch.zeros((1,)),
                    Y=torch.zeros((1, y_dim)),
                    opts=dict(
                        xlabel='Iteration',
                        ylabel='Loss',
                        title=args.save_folder.split('/')[-1],
                        legend=legend,
                    )
                )
            if use_refine:
                y_dis += [args.loss_coe[0]*loss_arm_l.cpu(),]
            # update = 'append' if iteration
            viz.line(
                X=torch.ones((1, y_dim)) * iteration,
                Y=torch.FloatTensor(y_dis).unsqueeze(0),
                win=lot,
                update='append',
                opts=dict(
                    xlabel='Iteration',
                    ylabel='Loss',
                    title=args.save_folder.split('/')[-1],
                    legend=legend,)
            )


    torch.save(ssd_net.state_dict(),
               os.path.join(args.save_folder, args.model_name + str(ssd_dim) + '_' + args.dataset_name + '_' +
                            repr(iteration) + '.pth'))
    print('Complet Training. Saving state, iter:', iteration)

# def adjust_learning_rate(optimizer, gamma):

   # for param_group in optimizer.param_groups:
   #     param_group['lr'] *= gamma

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):

     if epoch <= args.warm_epoch:
         lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warm_epoch)
     else:
         lr = args.lr * (gamma ** (step_index))
     for param_group in optimizer.param_groups:
         param_group['lr'] = lr
     # return lr

if __name__ == '__main__':
    train()
