import os

# from wandb.integration.sklearn.calculate import confusion_matrix

from utils.dataset import Loader, get_dataset
from utils.augment import EventAugment
import torch
from snn_utils.models import Classifier
import argparse
import wandb
import random
import numpy as np
from snn_utils.functions import TET_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torchviz import make_dot
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--l_mags', default=7, type=int,
                    help='Number of magnitudes')
parser.add_argument('--train_num_workers', default=0, type=int)
parser.add_argument('--train_batch_size', default=8, type=int)
parser.add_argument('--train_num_epochs', default=25, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument("--train_distributed", type=bool, default=False)
parser.add_argument("--augment_mode", default="identity",
                    choices=["identity", "RPG", "eventdrop", "NDA"])
parser.add_argument('--weight_decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--classifier', default="resnet18",
                    choices=["vgg11", "resnet34", "resnet18"])
parser.add_argument("--timesteps", default=10)
parser.add_argument("--seed", default=0)
parser.add_argument("--use_wandb", default=False)
parser.add_argument("--relcam_mask_std_ratio", default=1,
                    type=float, help="the std ratio of pre-mask in RPGMix")
parser.add_argument("--relcam_coord_std_ratio", default=0, type=float,
                    help="the std ratio of obtaining bounding boxes in RPGMix")
parser.add_argument('--spiking_neuron', default="PLIF",
                    choices=["LIF", "PLIF", "IF"])
parser.add_argument("--dataset", default="jester", choices=[
                    "DVSGesture", "jester"])
parser.add_argument("--relprop_mode", default="slrp",
                    type=str, choices=["slrp", "sltrp"])
parser.add_argument("--relevance_mix", default="layer4",
                    type=str, choices=["layer4", "long"])
parser.add_argument("--mask", default="post")
parser.add_argument("--mix_prob", default=0.5, type=float,
                    help="the max probability of mixing")
args = parser.parse_args()

if args.dataset == "DVSGesture":
    args.event_resolution = (128, 128)
    args.crop_dimension = (32, 32)
    args.num_classes = 11
    args.train_batch_size = 16
    args.timesteps = 16
    args.lr = 5e-4
    args.train_num_epochs = 25
elif args.dataset == "jester":
    args.event_resolution = (128, 128)
    args.crop_dimension = (64, 64)
    args.num_classes = 27
    args.train_batch_size = 8  # default 32
    args.timesteps = 16
    args.lr = 5e-4
    args.train_num_epochs = 25
else:
    raise Exception("Dataset not found")


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


if "resnet" in args.classifier:
    args.data_shape_prefix = "TB"
else:
    args.data_shape_prefix = "BT"

seed_everything(args.seed)
save_name = args.classifier + "_" + args.dataset + "_" + args.augment_mode
save_path = 'model/{}/SNN/{}'.format(args.dataset, save_name)
process_name = "SNN_" + save_name
save_path = 'model/{}/SNN/{}'.format(args.dataset, save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, "model_state_dict_cd_128_bs_8_direct.pth")
if args.use_wandb:
    wandb.init(project="EventRPG", name=process_name, config=args)

# ----------- CONFUSION MATRIX -----------#


def plot_confusion_matrix(true_labels, pred_labels, classes, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def mixup_criterion_raw_events(pred, target, tet_loss):
    if args.data_shape_prefix == "TB":  # T, B
        pred_mean = pred.mean(0)
    else:  # B, T
        pred_mean = pred.mean(1)
    sigma = target[:, 2]
    target = target[:, :2].to(torch.int64)
    loss = (sigma * tet_loss(pred, target[:, 0], mode=args.data_shape_prefix) + (
        1 - sigma) * tet_loss(pred, target[:, 1], mode=args.data_shape_prefix)).mean()
    accuracy = (pred_mean.argmax(1) == target[:, 0]).logical_or(
        pred_mean.argmax(1) == target[:, 1]).float().mean()
    return loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for events, labels, _ in val_loader:
            events, labels = events.to(device), labels.to(device)

            outputs = model(events)
            if args.data_shape_prefix == "TB":
                outputs_mean = outputs.mean(0)
            else:
                outputs_mean = outputs.mean(1)

            loss = criterion(outputs_mean, labels)
            total_loss += loss.item()
            total_correct += (outputs_mean.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


if __name__ == "__main__":
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = Classifier(voxel_dimension=(args.timesteps, *args.event_resolution), device=device, crop_dimension=args.crop_dimension,
                       classifier=args.classifier, num_classes=args.num_classes, pretrained=False, spiking_neuron=args.spiking_neuron,
                       relprop_mode=args.relprop_mode).to(device)
    event_augment = EventAugment(args.event_resolution, model, l_mags=args.l_mags,
                                 mask_std_ratio=args.relcam_mask_std_ratio,
                                 coord_std_ratio=args.relcam_coord_std_ratio, relevance_mix=args.relevance_mix,
                                 data_shape_prefix=args.data_shape_prefix, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_num_epochs, eta_min=0.)

    # Print the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    patience = 5
    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(args.train_num_epochs):
        train_ds, val_ds, test_ds = get_dataset(args)

        train_loader = Loader(
            train_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size, loader_type="")
        print("Snn train loader created...")

        test_loader = Loader(
            test_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size)
        print("Snn test loader created...")

        true_labels = []
        pred_labels = []
        print(f"Processing epoch: {epoch}")
        sum_accuracy = 0
        sum_loss = 0
        count = 1
        len_train_loader = len(train_loader)
        print("length of Training loader: ", len_train_loader)
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}", unit="batch") as pbar:
            for events, labels, folder_names in train_loader:
                # print(f"Labels: {labels}")
                # print(f"Folder name: {folder_names}")
                B = int(1 + events[-1, -1].item())
                spatial_mixup = False
                if args.augment_mode == 'RPG':
                    aug_events, labels, spatial_mixup = event_augment.batch_augment(
                        events, labels, args.mask, mix_prob=args.mix_prob)
                elif args.augment_mode == 'NDA':
                    aug_events = event_augment.nda_batch_augment(events)
                    if random.random() < 0.5:
                        spatial_mixup = True
                        aug_events, labels = event_augment.cut_mix(
                            aug_events, labels)
                elif args.augment_mode == 'eventdrop':
                    aug_events = event_augment.eventdrop_batch_augment(events)
                else:
                    aug_events = events

                model.train()
                optimizer.zero_grad()
                y = model(aug_events)

                if args.data_shape_prefix == "TB":  # T, B, C, H, W
                    y_mean = y.mean(0)
                    # print(f"student model output mean: {y_mean}")
                else:  # B, T, C, H, W
                    y_mean = y.mean(1)
                if spatial_mixup:
                    loss, accuracy = mixup_criterion_raw_events(
                        y, labels, TET_loss)
                else:
                    loss = TET_loss(y, labels, mode=args.data_shape_prefix)
                    accuracy = (y_mean.argmax(1) == labels).float().mean()
                sum_accuracy += accuracy
                sum_loss += loss
                loss.backward()
                optimizer.step()
                count += 1
                pbar.update(1)
        training_accuracy = sum_accuracy.item() / len(train_loader)
        training_loss = sum_loss.item() / len(train_loader)
        print(f"Epoch {str(epoch)}, Training Accuracy {
              str(training_accuracy)}")
        lr_scheduler.step()
        sum_accuracy = 0
        sum_loss = 0
        # -------------------- Validation --------------------#
        # val_loss, val_accuracy = validate(
        #     model, val_loader, cross_entropy_loss, device)
        # print(
        #     f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     early_stop_counter = 0
        #     torch.save(model.state_dict(), save_path)
        #     print("Model improved, saving best model.")
        # else:
        #     early_stop_counter += 1
        #     print(f"Early stopping counter: {early_stop_counter}/{patience}")

        # if early_stop_counter >= patience:
        #     print("Early stopping triggered. Training stopped.")
        #     break

        # ---------------------- Testing ----------------------#
        model.eval()
        sum_accuracy = 0
        sum_loss = 0
        for events, labels, folder_names in test_loader:
            # print(f"Labels: {labels}")
            with torch.no_grad():
                y = model(events)
                if args.data_shape_prefix == "TB":  # T, B, C, H, W
                    y_mean = y.mean(0)
                else:  # B, T, C, H, W
                    y_mean = y.mean(1)
                loss = cross_entropy_loss(y_mean, labels)
                accuracy = (y_mean.argmax(1) == labels).float().mean()

                # Confusion Matrix
                check_epoch = epoch

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(y_mean.argmax(1).cpu().numpy())

            sum_accuracy += accuracy
            sum_loss += loss
        test_loss = sum_loss.item() / len(test_loader)
        test_accuracy = sum_accuracy.item() / len(test_loader)
        print("Epoch {}, test Accuracy {}, test loss {}".format(
            str(epoch), str(test_accuracy), str(test_loss)))

        class_names = [
            "Doing other things",
            "Drumming Fingers",
            "No gesture",
            "Pulling Hand In",
            "Pulling Two Fingers In",
            "Pushing Hand Away",
            "Pushing Two Fingers Away",
            "Rolling Hand Backward",
            "Rolling Hand Forward",
            "Shaking Hand",
            "Sliding Two Fingers Down",
            "Sliding Two Fingers Left",
            "Sliding Two Fingers Right",
            "Sliding Two Fingers Up",
            "Stop Sign",
            "Swiping Down",
            "Swiping Left",
            "Swiping Right",
            "Swiping Up",
            "Thumb Down",
            "Thumb Up",
            "Turning Hand Clockwise",
            "Turning Hand Counterclockwise",
            "Zooming In With Full Hand",
            "Zooming In With Two Fingers",
            "Zooming Out With Full Hand",
            "Zooming Out With Two Fingers"
        ]
        save_path = f'CM_{epoch}.png'
        plot_confusion_matrix(true_labels, pred_labels,
                              class_names, save_path=save_path)

        if args.use_wandb:
            wandb.log({"training/accuracy": training_accuracy,
                       "training/loss": training_loss,
                       #    "validation/accuracy": val_accuracy,
                       #    "validation/loss": val_loss,
                       "test/accuracy": test_accuracy,
                       "test/loss": test_loss})

    torch.cuda.empty_cache()

# 0 - stop
# 1 - swiping_up
# 2 - swiping_down
# 3 - swiping_right
# 4 - swiping_left

# ANN
# Trainable Parameters: 11,070,203

# sew-resnet 18
# Trainable Parameters: 4,908,311

# in ANN
# 14 - stop             0
# 18 - swiping_up       1
# 15 - swiping_down     2
# 17 - swiping_right    3
# 16 - swiping_left     4
