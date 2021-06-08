import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F

from dataset import SequenceDataset
from model import transformer_fusion
from utils import ValueMeter, MeanTopKRecallMeter, topk_accuracy
from loss import SoftmaxEQL, get_ratio, SoftmaxEQL_Action

from tqdm import tqdm
from argparse import ArgumentParser
from os.path import join
import os
import numpy as np
import pandas as pd

parser = ArgumentParser(description="Training program for Mem")
parser.add_argument('--feat_in', type=int, default=1024,
                    help='Input size. If fusion, it is discarded (see --feats_in)')
parser.add_argument('--label', type=str, choices=['verb', 'noun', 'action', 'mix'], default='mix', )
parser.add_argument('--modality', type=str, choices=['rgb', 'flow', 'obj', 'fusion'], default='fusion')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seq_length', type=int, default=14)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--session', type=str, default='debug')
parser.add_argument('--root_dir', type=str, default='/root/Desktop/Data/Ego/')

args = parser.parse_args()
print(args)

path_to_models = 'checkpoint/' + args.session + 'len_' + str(
    args.seq_length) + '_modal_' + args.modality + '_label_' + args.label

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

Len_sequence = args.seq_length
Output_sequence = 4  # 1s
Input_sequence = Len_sequence - Output_sequence + 1  # +1 means including the timestamp of -1s

actions = pd.read_csv('actions.csv', index_col='id').to_numpy()
v_index = actions[:, 0].astype(int)
n_index = actions[:, 1].astype(int)


########################## Basic Functions #################################
def save_model(model, epoch, is_best_v=False, is_best_n=False, is_best_a=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                }, path_to_models + '.pth.tar')
    if is_best_v:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, join(
            path_to_models + '_best_verb.pth.tar'))
        print('save best verb model')

    if is_best_n:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, join(
            path_to_models + '_best_noun.pth.tar'))
        print('save best noun model')

    if is_best_a:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, join(
            path_to_models + '_best_action.pth.tar'))
        print('save best action model')


def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')

########################## Data #################################

if args.label == 'mix':
    args.label = ['verb', 'noun', 'action']

if args.modality == 'fusion':
    Dataset = SequenceDataset(path_to_lmdb=[args.root_dir + 'rgb',
                                            args.root_dir + 'flow',
                                            args.root_dir + 'obj'],
                              path_to_csv= args.root_dir + 'training.csv',
                              label_type=args.label, debug=args.debug,
                              sequence_length=Len_sequence)  # time_step=0.1, sequence_length=35, fps=30)

    Dataset_val = SequenceDataset(path_to_lmdb=[args.root_dir + 'rgb',
                                                args.root_dir + 'flow',
                                                args.root_dir + 'obj'],
                                  path_to_csv= args.root_dir + 'validation.csv',
                                  label_type=args.label, debug=args.debug,
                                  sequence_length=Len_sequence)  # time_step=0.1, sequence_length=35, fps=30)
else:
    Dataset = SequenceDataset(path_to_lmdb=[args.root_dir  + args.modality],
                              path_to_csv=args.root_dir + 'training.csv',
                              label_type=args.label, debug=args.debug,
                              sequence_length=Len_sequence)  # time_step=0.1, sequence_length=35, fps=30)

    Dataset_val = SequenceDataset(path_to_lmdb=[args.root_dir+ args.modality],
                                  path_to_csv=args.root_dir + 'validation.csv',
                                  label_type=args.label, debug=args.debug,
                                  sequence_length=Len_sequence)  # time_step=0.1, sequence_length=35, fps=30)

Dataloader = data.DataLoader(Dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
Dataloader_val = data.DataLoader(Dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

loaders = {'training': Dataloader, 'validation': Dataloader_val}
labels = np.stack(Dataset.labels)

if args.debug is True:
    class_num_v = 97
    class_num_n = 300
    class_num = 3806
else:
    class_num_v = max(labels[:, 0]) + 1
    class_num_n = max(labels[:, 1]) + 1
    class_num = max(labels[:, 2]) + 1

########################## Model and Criterion #################################

model_c = transformer_fusion(seq_len=Input_sequence, num_class_noun=class_num_n, num_class_verb=class_num_v,
                             num_class_action=class_num, feat_in=1024, hidden=1024, ).cuda()

Criterion = torch.nn.MSELoss()
Criterion_SEQL_verb = SoftmaxEQL(labels=labels[:, 0])
Criterion_SEQL_noun = SoftmaxEQL(labels=labels[:, 1])
Criterion_SEQL_action = SoftmaxEQL_Action(Criterion_SEQL_verb.class_weight.detach(),
                                          Criterion_SEQL_noun.class_weight.detach(),
                                          torch.from_numpy(v_index).cuda(), torch.from_numpy(n_index).cuda())

Optimizer_c = torch.optim.SGD(model_c.parameters(), lr=0.01, momentum=0.9)


##########################   Training    #################################

device = torch.device('cuda')
dtype = torch.float32

best_perf_n = 0
best_perf_v = 0
best_perf_a = 0

for epoch in range(100):
    loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
    accuracy_meter_v = {'training': MeanTopKRecallMeter(class_num_v),
                        'validation': MeanTopKRecallMeter(class_num_v)}
    accuracy_meter_n = {'training': MeanTopKRecallMeter(class_num_n),
                        'validation': MeanTopKRecallMeter(class_num_n)}

    accuracy_meter_v_1 = {'training': MeanTopKRecallMeter(class_num_v),
                          'validation': MeanTopKRecallMeter(class_num_v)}
    accuracy_meter_n_1 = {'training': MeanTopKRecallMeter(class_num_n),
                          'validation': MeanTopKRecallMeter(class_num_n)}

    accuracy_meter_a = {'training': MeanTopKRecallMeter(class_num),
                        'validation': MeanTopKRecallMeter(class_num)}

    accuracy_meter_a_1 = {'training': MeanTopKRecallMeter(class_num),
                          'validation': MeanTopKRecallMeter(class_num)}

    for mode in ['training', 'validation']:
        loss_c_all = 0
        batch_i = 1

        if mode == 'training':
            model_c.train()
        else:
            model_c.eval()

        with tqdm(loaders[mode], unit="batch") as tepoch:
            for batch in tepoch:
                feature_1 = batch['past_features'][0]
                feature_2 = batch['past_features'][1]
                feature_3 = batch['past_features'][2]

                encode_input_1 = feature_1[:, :Input_sequence, :].to(device, dtype)  # rgb
                encode_input_2 = feature_2[:, :Input_sequence, :].to(device, dtype)  # flow
                encode_input_3 = feature_3[:, :Input_sequence, :].to(device, dtype)  # obj

                y = batch['label'].to(device)

                if mode == 'training':
                    preds_v, preds_n, preds_a, preds_v_1, preds_n_1, preds_a_1 = model_c(
                        encode_input_1, encode_input_2, encode_input_3)

                else:
                    with torch.no_grad():
                        preds_v, preds_n, preds_a, preds_v_1, preds_n_1, preds_a_1 = model_c(
                            encode_input_1, encode_input_2, encode_input_3)

                linear_preds_v = preds_v.view(-1, preds_v.shape[-1])
                linear_preds_v_1 = preds_v_1.view(-1, preds_v_1.shape[-1])
                linear_labels_v = y[:, 0]

                linear_preds_n = preds_n.view(-1, preds_n.shape[-1])
                linear_preds_n_1 = preds_n_1.view(-1, preds_n_1.shape[-1])
                linear_labels_n = y[:, 1]

                linear_preds_action = preds_a.view(-1, preds_a.shape[-1])
                linear_preds_action_1 = preds_a_1.view(-1, preds_a_1.shape[-1])
                linear_labels_action = y[:, 2]

                loss_c_v = Criterion_SEQL_verb(linear_preds_v, linear_labels_v) \
                           + 0.5 * Criterion_SEQL_verb(linear_preds_v_1, linear_labels_v)

                loss_c_n = Criterion_SEQL_noun(linear_preds_n, linear_labels_n) \
                           + 0.5 * Criterion_SEQL_noun(linear_preds_n_1, linear_labels_n)

                loss_c_a = Criterion_SEQL_action(linear_preds_action, linear_labels_action) \
                           + 0.5 * Criterion_SEQL_action(linear_preds_action_1, linear_labels_action)

                # if epoch>=6:
                #     loss_c = 0.1*loss_c_v + 0.1*loss_c_n + loss_c_a
                # else:
                loss_c = 0.5 * loss_c_v + 0.5 * loss_c_n + loss_c_a
                # l2 loss
                # loss_f = Criterion(pred_output, decode_output)

                # contrastive loss
                # logits, labels = contrast_loss(pred_output.reshape(-1,1024), decode_output.reshape(-1,1024))
                # loss_f = F.cross_entropy(logits, labels)

                if mode == 'training':
                    Optimizer_c.zero_grad()
                    loss = loss_c
                    loss.backward()

                    Optimizer_c.step()
                else:
                    loss = loss_c

                loss_c_all += loss_c.item()

                # use top-5 for anticipation
                k = 5

                acc_v = topk_accuracy(
                    preds_v.detach().cpu().numpy(), linear_labels_v.detach().cpu().numpy(), (k,))[0] * 100
                acc_n = topk_accuracy(
                    preds_n.detach().cpu().numpy(), linear_labels_n.detach().cpu().numpy(), (k,))[0] * 100
                acc_a = topk_accuracy(
                    linear_preds_action.detach().cpu().numpy(), linear_labels_action.detach().cpu().numpy(), (k,))[
                            0] * 100

                loss_meter[mode].add(loss.item(), preds_n_1.shape[0])
                accuracy_meter_v[mode].add(preds_v.detach().cpu().numpy(),
                                           linear_labels_v.detach().cpu().numpy())
                accuracy_meter_v_1[mode].add(preds_v_1.detach().cpu().numpy(),
                                             linear_labels_v.detach().cpu().numpy())

                accuracy_meter_n[mode].add(preds_n.detach().cpu().numpy(),
                                           linear_labels_n.detach().cpu().numpy())
                accuracy_meter_n_1[mode].add(preds_n_1.detach().cpu().numpy(),
                                             linear_labels_n.detach().cpu().numpy())

                accuracy_meter_a[mode].add(linear_preds_action.detach().cpu().numpy(),
                                           linear_labels_action.detach().cpu().numpy())

                accuracy_meter_a_1[mode].add(linear_preds_action_1.detach().cpu().numpy(),
                                             linear_labels_action.detach().cpu().numpy())

                tepoch.set_postfix(loss_c=loss_c_all / batch_i, loss_v=loss_c_v.item(), loss_n=loss_c_n.item(),
                                   loss_a=loss_c_a.item())
                # print('loss: {:.4e}'.format(loss))
                # if batch_i % 100 == 0:
                #     print('loss: {:.4e}'.format(loss))

                e = epoch + batch_i / len(loaders[mode])

                batch_i += 1

            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_v[mode],
                max(accuracy_meter_v[mode].value(), best_perf_v) if mode == 'validation'
                else None, green=True)
            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_v_1[mode],
                max(accuracy_meter_v_1[mode].value(), best_perf_v) if mode == 'validation'
                else None, green=True)

            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_n[mode],
                max(accuracy_meter_n[mode].value(), best_perf_n) if mode == 'validation'
                else None, green=True)
            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_n_1[mode],
                max(accuracy_meter_n_1[mode].value(), best_perf_n) if mode == 'validation'
                else None, green=True)

            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_a[mode],
                max(accuracy_meter_a[mode].value(), best_perf_a) if mode == 'validation'
                else None, green=True)
            log(mode, epoch + 1, loss_meter[mode], accuracy_meter_a_1[mode],
                max(accuracy_meter_a_1[mode].value(), best_perf_a) if mode == 'validation'
                else None, green=True)

    if best_perf_v < accuracy_meter_v['validation'].value():
        best_perf_v = accuracy_meter_v['validation'].value()
        is_best_v = True
    else:
        is_best_v = False

    if best_perf_n < accuracy_meter_n['validation'].value():
        best_perf_n = accuracy_meter_n['validation'].value()
        is_best_n = True
    else:
        is_best_n = False

    if best_perf_a < accuracy_meter_a['validation'].value():
        best_perf_a = accuracy_meter_a['validation'].value()
        is_best_a = True
    else:
        is_best_a = False

    save_model(model_c, epoch + 1, is_best_v=is_best_v, is_best_n=is_best_n, is_best_a=is_best_a)
