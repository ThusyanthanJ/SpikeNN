if __name__ == '__main__':
    import time
    from utils.dataset import Loader, get_dataset
    from configuration import settings
    from tqdm import tqdm
    import random
    import torch
    import argparse
    import torchvision
    from transforms import *
    from models import TSN
    from ann_dataset import TSNDataSet
    import datasets_video
    import os
    from snn_utils.models import Classifier
    from utils.augment import EventAugment
    from snn_utils.functions import TET_loss
    from itertools import zip_longest
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument('--l_mags', default=7, type=int,
                        help='Number of magnitudes')
    parser.add_argument('--train_num_workers', default=4, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--train_num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument("--train_distributed", type=bool, default=False)
    parser.add_argument("--augment_mode", default="identity",
                        choices=["identity", "RPG", "eventdrop", "NDA"])
    parser.add_argument('--weight_decay', '--wd', default=0,
                        type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--classifier', default="resnet18",
                        choices=["vgg11", "resnet34", "resnet18"])
    parser.add_argument("--timesteps", default=16)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--max_num", type=int, default=-1)
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument("--relcam_mask_std_ratio", default=1,
                        type=float, help="the std ratio of pre-mask in RPGMix")
    parser.add_argument("--relcam_coord_std_ratio", default=0, type=float,
                        help="the std ratio of obtaining bounding boxes in RPGMix")
    parser.add_argument('--spiking_neuron', default="PLIF",
                        choices=["LIF", "PLIF", "IF"])
    parser.add_argument("--dataset", default="jester", choices=[
                        "DVSGesture", "Jester"])
    parser.add_argument("--relprop_mode", default="slrp",
                        type=str, choices=["slrp", "sltrp"])
    parser.add_argument("--relevance_mix", default="layer4",
                        type=str, choices=["layer4", "long"])
    parser.add_argument("--mask", default="post")
    parser.add_argument("--mix_prob", default=0.5, type=float,
                        help="the max probability of mixing")
    parser.add_argument("--root_path", default="")
    parser.add_argument('--weights', type=str,
                        default='pretrained_models/MFF_jester_RGBFlow_BNInception_segment4_3f1c_best.pth.tar')
    parser.add_argument('--num_motion', type=int, default=3)

    args = parser.parse_args()
    args.event_resolution = (128, 128)
    args.crop_dimension = (64, 64)
    args.num_classes = 5
    soft_target_loss_weight = 0.25
    cross_entropy_loss_weight = 0.75

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU instead.")

    ########################################################### SNN ###########################################################

    def plot_confusion_matrix(true_labels, pred_labels, classes, save_path):
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix- Knowledge distillation")
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

    ###########################################################################################################################
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
                total_correct += (outputs_mean.argmax(1)
                                  == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    ########################################################### ANN ###########################################################

    dataset = 'jester'
    test_segments = 4
    modality = 'RGBFlow'
    arch = 'BNInception'
    consensus_type = 'MLP'
    img_feature_dim = 256
    temperature = 4
    save_path_kd = "model/SNN_KD/"

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(
        args.dataset, modality)
    # print(f"Categories: {categories}")

    teacher_ann_net = TSN(num_class=27, num_segments=test_segments, modality=modality,
                          base_model=arch, consensus_type=consensus_type, img_feature_dim=img_feature_dim)

    print("Loading pretrained weights...")
    checkpoint = torch.load(args.weights, map_location=device)
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
        checkpoint['state_dict'].items())}

    try:
        teacher_ann_net.load_state_dict(base_dict)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

    teacher_ann_net = teacher_ann_net.to(device)

    num_motion = 3
    cropping = torchvision.transforms.Compose([
        GroupScale(teacher_ann_net.scale_size),
        GroupCenterCrop(teacher_ann_net.input_size),
    ])
    data_length = args.num_motion

    ###########################################################################################################################

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    student_snn_net = Classifier(voxel_dimension=(args.timesteps, *args.event_resolution), device=device, crop_dimension=args.crop_dimension,
                                 classifier=args.classifier, num_classes=args.num_classes, pretrained=False, spiking_neuron=args.spiking_neuron,
                                 relprop_mode=args.relprop_mode).to(device)
    event_augment = EventAugment(args.event_resolution, student_snn_net, l_mags=args.l_mags,
                                 mask_std_ratio=args.relcam_mask_std_ratio,
                                 coord_std_ratio=args.relcam_coord_std_ratio, relevance_mix=args.relevance_mix,
                                 data_shape_prefix=args.data_shape_prefix, device=device)

    optimizer = torch.optim.Adam(
        student_snn_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_num_epochs, eta_min=0.)
    best_val_accuracy = 0

    def eval_video(video_data):
        i, data, label = video_data

        try:
            data = data.to(device)
            length = 3 + 2 * num_motion

            with torch.no_grad():
                input_var = data.view(-1, length, data.size(2), data.size(3))
                rst = teacher_ann_net(input_var)

            rst_tensor = rst.clone()
            rst = rst.cpu().numpy()

            rst = rst.reshape(-1, 1, 27)

            return rst, rst_tensor
        except Exception as e:
            print(f"Error in eval_video: {e}")
            raise

    patience = 5
    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(args.train_num_epochs):

        train_ds, val_ds, test_ds = get_dataset(args)
        snn_train_loader = Loader(
            train_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size, loader_type="train_ds")
        print("Snn train loader created...")

        snn_test_loader = Loader(
            test_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size)
        print("Snn test loader created...")

        snn_val_loader = Loader(
            val_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size
        )

        ann_data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=test_segments,
                       new_length=data_length,
                       modality=modality,
                       image_tmpl=prefix,
                       dataset=args.dataset,
                       test_mode=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(arch in ['BNInception', 'InceptionV3']), isRGBFlow=(
                               modality == 'RGBFlow')),
                           ToTorchFormatTensor(
                               div=(arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(teacher_ann_net.input_mean,
                                          teacher_ann_net.input_std),
                       ])),
            batch_size=args.train_batch_size, shuffle=False,
            num_workers=args.workers*2, pin_memory=True
        )
        print("Ann test loader created...")

        settings.IS_TRAIN = True
        true_labels = []
        pred_labels = []
        running_loss = 0.0
        start = time.time()
        student_snn_net.train()
        teacher_ann_net.eval()
        correct = 0.0
        num_sample = 0
        sum_accuracy = 0
        sum_loss = 0
        count = 1
        teacher_output = []
        max_num = args.max_num if args.max_num > 0 else len(
            ann_data_loader.dataset)

        total_num = min(max_num, len(ann_data_loader.dataset))
        with tqdm(total=len(snn_train_loader), desc=f"Training Epoch {epoch}", unit="batch") as pbar:
            for (events, labels, folder_names), (i, (data, label)) in zip(snn_train_loader, enumerate(ann_data_loader)):

                # rgb_batch = bacth from
                ############################# SNN Train #########################
                spatial_mixup = False
                aug_events = events
                y = student_snn_net(aug_events)
                if args.data_shape_prefix == "TB":
                    y_mean = y.mean(0)
                else:
                    y_mean = y.mean(1)

                if spatial_mixup:
                    loss, accuracy = mixup_criterion_raw_events(
                        y, labels, TET_loss)
                else:
                    loss = TET_loss(y, labels, mode=args.data_shape_prefix)
                    accuracy = (y_mean.argmax(1) == labels).float().mean()

                #################################################################

                ############################# ANN Train #########################

                if i*4 >= max_num:
                    break
                try:
                    rst, rst_tensor = eval_video(
                        (i, data, label))
                    new_teacher_output = []
                    for j in range(len(data)):
                        indices = np.array([1, 19, 20, 9, 23])
                        element = rst[j][0][indices]
                        # element = rst[j][0][14:19]
                        new_teacher_output.append(element)
                    new_teacher_output_tensor = torch.tensor(
                        np.array(new_teacher_output))

                except Exception as e:
                    print(f"Error in processing rgb video {i}: {e}")
                    continue

                # print("")
                # print(f"student output: {y_mean}")
                # print(f"teacher output : {new_teacher_output_tensor}")

                student_logits = y_mean
                teacher_logits = new_teacher_output_tensor

                soft_targets = nn.functional.softmax(
                    teacher_logits / temperature, dim=1
                ).to(device=device)

                soft_prob = nn.functional.softmax(
                    student_logits / temperature, dim=1
                ).to(device=device)

                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                    temperature ** 2)

                label_loss = cross_entropy_loss(student_logits, labels)

                loss = soft_target_loss_weight * soft_targets_loss + \
                    cross_entropy_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                sum_accuracy += accuracy
                sum_loss += loss

                count += 1
                pbar.update(1)

        training_accuracy = sum_accuracy.item() / len(snn_train_loader)
        training_loss = sum_loss.item() / len(snn_train_loader)
        print(f"Epoch {str(epoch)}, Training Accuracy {
              str(training_accuracy)}")
        lr_scheduler.step()
        sum_accuracy = 0
        sum_loss = 0
        # -------------------- Validation --------------------#
        val_loss, val_accuracy = validate(
            student_snn_net, snn_val_loader, cross_entropy_loss, device)
        print(
            f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            # torch.save(student_snn_net.state_dict(), save_path_kd)
            print("Model improved")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
        ########################### Testing Snn ###################################
        student_snn_net.eval()

        sum_accuracy = 0
        sum_loss = 0
        for events, labels, folder_names in snn_test_loader:
            with torch.no_grad():
                y = student_snn_net(events)
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
        test_loss = sum_loss.item() / len(snn_test_loader)
        test_accuracy = sum_accuracy.item() / len(snn_test_loader)
        print("Epoch {}, test Accuracy {}, test loss {}".format(
            str(epoch), str(test_accuracy), str(test_loss)))

        class_names = ['drumming fingers',
                       'thumb down',
                       'thumb up',
                       'shaking hand',
                       'zooming in with full hand'
                       ]
        save_path = f'KD_CM_{epoch}.png'
        plot_confusion_matrix(true_labels, pred_labels,
                              class_names, save_path=save_path)
