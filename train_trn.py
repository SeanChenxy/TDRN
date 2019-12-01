import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AnnotationTransform, BaseTransform, VOCDetection, detection_collate, pair_collate, mb_cfg, dataset_training_cfg
from utils.augmentations import SSDAugmentation, pairSSDAugmentation
from layers.modules import RefineMultiBoxLoss
from layers.functions import PriorBox
from model.networks import net_init
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
    logging.info('Deform: ' + str(args.deform))
    logging.info('loose: ' + str(args.loose))
    if args.resume_static:
        logging.info('resume_static: '+ args.resume_static )
    if args.resume:
        logging.info('resume: '+ args.resume )
    logging.info('start_iter: '+ str(args.start_iter))
    logging.info('lr: '+ str(args.lr))
    logging.info('gamam: '+ str(args.gamma))
    logging.info('step_list: '+ str(args.step_list))
    logging.info('save_interval: '+ str(args.save_interval))
    logging.info('dataset_name: '+ args.dataset_name )
    logging.info('set_file_name: '+ args.set_file_name )
    logging.info('gpu_ids: '+ args.gpu_ids)
    logging.info('augm_type: '+ args.augm_type)
    logging.info('batch_size: '+ str(args.batch_size))
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--basenet', default='vgg16bn_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--resume_static', default='./weights040/TRN/SSD320VggBn_VIDDET/ssd320_VIDDET_100.pth', type=str, help='Resume from checkpoint')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='./weights040/test', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='VIDDET', help='VOC0712/VIDDET/seqVID2017/MOT17Det/seqMOT17Det')
parser.add_argument('--step_list', nargs='+', type=int, default=[30,50], help='step_list for learning rate')
parser.add_argument('--backbone', default='VGG', type=str, help='Backbone')
parser.add_argument('--c7_channel', default=1024, type=int, help='out_channel of Conv7 in VGG')
parser.add_argument('--deform', default=True, type=str2bool, help='use defomable conv for detection.')
parser.add_argument('--loose', default=1., type=float, help='loose reference')
parser.add_argument('--model_name', default='trn', type=str, help='which model selected')
parser.add_argument('--ssd_dim', default=320, type=int, help='ssd_dim 300, 320 or 512')
parser.add_argument('--gpu_ids', default='4,5', type=str, help='gpu number')
parser.add_argument('--augm_type', default='ssd', type=str, help='how to transform data')
parser.add_argument('--set_file_name',  default='train_VID_DET', type=str, help='train_VID_DET/train_video_remove_no_object/train, MOT dataset does not use it')
parser.add_argument('--bn', default=True, type=str2bool, help='select sequence data in a skip way')
parser.add_argument('--save_interval', default=10, type=int, help='frequency of checkpoint saving')
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

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

ssd_dim = args.ssd_dim
prior = 'VOC_'+ str(args.ssd_dim)
cfg = mb_cfg[prior]
train_sets, num_classes, data_root = dataset_training_cfg[args.dataset_name]
logging.info('train sets: ' + str(train_sets))
set_filename = args.set_file_name

means = (104, 117, 123)
mean_np = np.array(means, dtype=np.int32)
batch_size = args.batch_size
weight_decay = args.weight_decay
max_epoch = args.step_list[-1]
gamma = args.gamma
momentum = args.momentum
if args.augm_type == 'ssd':
    data_transform = SSDAugmentation
    collate_fn = detection_collate
elif args.augm_type == 'pairssd':
    data_transform = pairSSDAugmentation
    collate_fn = pair_collate
else:
    data_transform = BaseTransform
    collate_fn = detection_collate
if args.visdom:
    import visdom
    viz = visdom.Visdom()

if 'Mob' in args.backbone:
    from model.ssd4scale_mobile import build_net
    static = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel)
    ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel, deform=args.deform)
    init_option = 'VGG4s'
else:
    from model.ssd4scale_vgg import build_net
    static = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel, bn=args.bn)
    ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, c7_channel=args.c7_channel, bn=args.bn, deform=args.deform)
    init_option = 'VGG4s'

# net = ssd_net
static.load_weights(args.resume_static)
static.eval()
static_net = torch.nn.DataParallel(static).to(device)
net = torch.nn.DataParallel(ssd_net).to(device)

if args.resume:
    logging.info('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    backbone_weights = torch.load('../weights/'+ args.basenet)
    logging.info('Loading base network...')
    ssd_net.backbone.load_state_dict(backbone_weights)
    net_init(ssd_net, init_option, logging, deform=args.deform)


tarin_list = net.parameters()
optimizer = optim.SGD(tarin_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, device=device)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward().to(device)

def train():
    net.train()
    epoch = args.start_iter
    dataset = VOCDetection(data_root, train_sets, data_transform(ssd_dim, means),
                           AnnotationTransform(dataset_name=args.dataset_name),
                           dataset_name=args.dataset_name, set_file_name=set_filename)

    epoch_size = len(dataset) // args.batch_size
    drop_step = [s * epoch_size for s in args.step_list]
    max_iter = max_epoch * epoch_size
    logging.info('Loading Dataset:' + args.dataset_name + ' dataset size: ' +str(len(dataset)))

    step_index = 0
    if args.visdom:
        legend = ['l', 'c']
        y_dim = len(legend)
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
            batch_iterator = iter(data_loader)
            if epoch % args.save_interval == 0:
                logging.info('Saving state, epoch: '+ str(epoch))
                torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.model_name + str(
                    ssd_dim) + '_' + args.dataset_name + '_' +repr(epoch) + '.pth'))
            epoch += 1

        t0 = time.time()
        if iteration in drop_step:
            step_index = drop_step.index(iteration) + 1
        adjust_learning_rate(optimizer, args.gamma, step_index)
        if args.augm_type == 'pairssd':
            images_ori, images_trans, _, targets = next(batch_iterator)
            with torch.no_grad():
                images_ori = images_ori.to(device)
                images_trans = images_trans.to(device)
                targets = [anno.to(device) for anno in targets]
            static_out = static_net(images_ori)
            out = net(images_trans)
        else:
            images, targets = next(batch_iterator)
            with torch.no_grad():
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]
            static_out = list(static_net(images, ret_loc=args.deform))
            static_out[0] *= args.loose
            if args.deform:
                ref_loc = static_out[2]
            else:
                ref_loc = list()
            out = net(images, ref_loc=ref_loc)
        # backward
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets, arm_data=static_out[:2])
        loss =loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % 10 == 0:
            logging.info('Epoch:' + repr(epoch) + ', epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + ', total_iter ' + repr(
                iteration) + ' || loss: %.2f, Loss_l: %.2f, loss_c: %.2f, lr: %.5f || Timer: %.4f sec.' % (
                         loss, loss_l, loss_c, optimizer.param_groups[0]['lr'], t1 - t0))
        if args.visdom:
            y_dis = [loss_l.cpu(), loss_c.cpu(),]
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


def adjust_learning_rate(optimizer, gamma, step_index):

     lr = args.lr * (gamma ** (step_index))
     for param_group in optimizer.param_groups:
         param_group['lr'] = lr
     # return lr

if __name__ == '__main__':
    train()
