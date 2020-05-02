# !/usr/bin/env python3
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import argparse
from yolo_model.models_storage import get_readable_timestamp
from yolo_model.yolo import YoloDetectorV1, YoloDetectorV1MobNet
from yolo_model.loss import yolo_v1_loss
from pascal_voc.voc_detection import VocDetection, decode_output_tensor, \
    decode_output_tensor_act, get_prediction_img
from utils.img_proc import *
import torch
from yolo_model.models_storage import save_model, load_model, get_default_model_name
from torch.utils.tensorboard import SummaryWriter
import time
from training.lr_scheduler import *
from metrics.detection_metrics import *
from os.path import isfile, isdir


class Logger:

    def __init__(self, logfile=None):
        self.logfile = None
        self.set_log_path(logfile)
        pass

    def set_log_path(self, path):
        self.logfile = path
        pass

    def log(self, *msg):
        info = "\n" + get_readable_timestamp() + " <DetectorTrainer>: "
        for m in msg:
            info += m + " "
        print(info)
        info += "\n"
        if self.logfile is not None:
            with open(self.logfile, "a") as f:
                f.write(info)
    pass


def get_model(args, logger):
    if args.model:
        if args.model == "YOLOv1":
            if args.pretrained != "no":
                # yolo1 pretrained
                if args.checkpoint:
                    # load yolo1 from checkpoint
                    logger.log("Model (YOLOv1) loaded from checkpoint"+str(args.checkpoint))
                    return load_model(args.checkpoint)
                else:
                    # model with pretrained feature extractor
                    logger.log("Model (YOLOv1) loaded with pretrained feature extractor")
                    return YoloDetectorV1(pretrained=True)
            else:
                # yolo with random pars
                return YoloDetectorV1(pretrained=False)
        elif args.model == "YOLOv1MobNet2":
            if args.pretrained != "no":
                # yolo1 pretrained
                if args.checkpoint:
                    # load yolo1 from checkpoint
                    logger.log("Model (YOLOv1 with Mobilenet_v2) loaded from checkpoint"+str(args.checkpoint))
                    return load_model(args.checkpoint)
                else:
                    # model with pretrained feature extractor
                    logger.log("Model (YOLOv1 with Mobilenet_v2) loaded with pretrained feature extractor")
                    return YoloDetectorV1MobNet(pretrained=True)
            else:
                # yolo with random pars
                return YoloDetectorV1MobNet(pretrained=False)
        else:
            logger.log("Unknown model selected: " + args.model + ". Terminating...")
            exit(1)
    else:
        # model with pretrained feature extractor
        logger.log("Loaded default model: YOLOv1 with pretrained MobileNetv2 feature extractor")
        def_model_path = get_default_model_name()
        if def_model_path == "":
            logger.log("Default YOLOv1 with pretrained MobileNetv2 fext and untrained head")
            return YoloDetectorV1MobNet(pretrained=True)
        else:
            logger.log("YOLOv1 with MobileNetv2 from the checkpoint: ", def_model_path)
            return load_model(def_model_path)


def get_dataset(args, logger, path):
    if args.dataset:
        if args.dataset == "VOC":
            logger.log("Loaded VOC dataset from ", path)
            return VocDetection(path, subset="trainval", target_shape=(224, 224), size_limit=args.dset_size_limit)
        else:
            logger.log("Unknown dataset selected: " + args.dataset + ". Terminating...")
            exit(1)
    else:
        logger.log("Loaded default dataset: Pascal VOC for detection from ", path)
        return VocDetection(path, subset="trainval", target_shape=(224, 224), size_limit=args.dset_size_limit)


def get_dataset_root():
    # pascal_voc_root = "/home/igor/datasets/VOC_2007/"
    pascal_voc_root = "/home/igor/datasets/VOC_2007/"
    # pascal_voc_root = "/home/igor/datasets/VOC_merged/"
    dataset_path_ini = "dataset_path.txt"
    if isfile(dataset_path_ini):
        with open(dataset_path_ini, "r") as f:
            line = f.readline()
            if line:
                line = line.replace("\n", "")
                if isdir(line):
                    return line
    return pascal_voc_root


def get_training_config(args):
    use_cuda = True
    epochs_limit = 10
    train_batch_size = 8
    valid_batch_size = 4
    autosave_period_epochs = None
    autosave_period_batches = None
    valid_period = 25
    default_lr = 1e-3
    path_to_schedule = "lr_schedule.txt"

    if args.use_cuda:
        use_cuda = bool(int(args.use_cuda))
    if args.epochs:
        epochs_limit = int(args.epochs)
    if args.batch_size:
        train_batch_size = int(args.batch_size)
    if args.batch_size_val:
        valid_batch_size = int(args.batch_size_val)
    if args.autosave_period_epochs:
        autosave_period_epochs = int(args.autosave_period_epochs)
    if args.autosave_period_batches:
        autosave_period_batches = int(args.autosave_period_batches)
    if args.valid_period:
        valid_period = int(args.valid_period)

    out = (use_cuda, epochs_limit, train_batch_size, valid_batch_size, autosave_period_epochs, autosave_period_batches,
           valid_period, default_lr, path_to_schedule)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of target model class (YOLOv1, YOLOv2 etc.)")
    parser.add_argument("--pretrained", help="Pretrained model or no")
    parser.add_argument("--checkpoint", help="Absolute path to saved torch model")
    parser.add_argument("--dataset", help="Name of dataset for training (VOC2007, VOC2012 etc)")
    parser.add_argument("--epochs", help="Count of epochs to learn")
    parser.add_argument("--batch_size", help="Batch size for training")
    parser.add_argument("--batch_size_val", help="Batch size for validation")
    parser.add_argument("--autosave_period_epochs", help="Period of auto save in epochs")
    parser.add_argument("--autosave_period_batches", help="Autosave period in batches")
    parser.add_argument("--valid_period", help="Period for auto validation (testing)")
    parser.add_argument("--log_path", help="Abs path to log file")
    parser.add_argument("--use_cuda", help="Use GPU (if available)")
    parser.add_argument("--lr", help="Learning rate for non-scheduled (default) training")
    parser.add_argument("--lr_schedule_path", help="Path to file with LR schedule for training")
    parser.add_argument("--dset_size_limit", help="Limit for dataset size (in images)")
    args = parser.parse_args()

    logger = Logger()
    # if args.log_path:
    #     logger.set_log_path(args.log_path);
    logger.set_log_path("log.txt")

    logger.log("\n\n\n==================== Training session ====================")
    logger.log("Start...")

    use_cuda, epochs_cnt, train_batch_size, val_batch_size, autosave_period_epochs, autosave_period_batches, \
    valid_period, lr, schedule_path = get_training_config(args)

    CUDA_IS_AVAILABLE = torch.cuda.is_available() if use_cuda else False
    processor_device_name = 'cuda' if CUDA_IS_AVAILABLE else 'cpu'
    processor_device = torch.device(processor_device_name)
    logger.log("GPU availability: ", str(torch.cuda.is_available()), " Enabled: ", str(use_cuda))
    logger.log("Processing device: ", processor_device_name)
    logger.log("Device object: ", str(processor_device))

    model = get_model(args, logger)

    dataset_root = get_dataset_root()
    trainval_dataset = get_dataset(args, logger, dataset_root)

    validation_split = .1
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(trainval_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(trainval_dataset, batch_size=train_batch_size,
                                                   sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(trainval_dataset, batch_size=val_batch_size,
                                                        sampler=valid_sampler)

    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,  shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # optim.Adam(model.parameters(), lr=learning_rate)
    batches_per_epochs = len(trainval_dataset) * (1. - validation_split) / train_batch_size

    schedule_plan = load_default_schedule_plan(batches_per_epochs, path=schedule_path, def_lr=lr)
    lr_scheduler = LRScheduler(schedule_plan)
    scheduled_optimizer = OptimizerWithSchedule(optimizer, lr_scheduler)

    tboard_writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)

    global_step_cnt = 0

    if CUDA_IS_AVAILABLE:
        model.cuda()

    logger.log("Training on GPU: ", str(CUDA_IS_AVAILABLE))
    logger.log("Model: ", str(model))
    logger.log("Dataset: ", str(trainval_dataset))
    logger.log("Epochs count: ", str(epochs_cnt))
    logger.log("Batch size (train): ", str(train_batch_size))
    logger.log("Batch size (val): ", str(val_batch_size))
    logger.log("Validation period: ", str(valid_period))
    logger.log("Optimizer: ", str(optimizer))
    logger.log("Autosave period (ep.): ", str(autosave_period_epochs))
    logger.log("Autosave period (b.): ", str(autosave_period_batches))
    logger.log("LR schedule: ")
    for idx, stage in enumerate(lr_scheduler.schedule):
        logger.log("Stage ", str(idx), " Parameters: ", str(stage))
    parameters_list = [str(w.shape) for w in list(model.parameters())]
    parameters_text = ""
    for p in parameters_list:
        parameters_text += "\n" + p
    logger.log("Model parameters for optimization: ", parameters_text)

    for e in range(epochs_cnt):
        for batch_ndx, batch in enumerate(train_dataloader):
            iter_start_timastamp = time.time()

            # optimizer.zero_grad()
            scheduled_optimizer.zero_grad()

            image_batch = batch["input"]
            target_batch = batch["target"]

            if CUDA_IS_AVAILABLE:
                image_batch = image_batch.to(device=processor_device)
                target_batch = target_batch.to(device=processor_device)

            model.train()

            prediction_batch = model.forward(image_batch)

            box_regr_loss, pos_conf_loss, neg_conf_loss, probs_loss = yolo_v1_loss(prediction_batch, target_batch)
            total_loss = box_regr_loss + pos_conf_loss + neg_conf_loss + probs_loss

            total_loss.backward()
            # optimizer.step()
            scheduled_optimizer.step()

            iter_stop_timastamp = time.time()
            iteration_duration = iter_stop_timastamp - iter_start_timastamp

            cur_state = "\n"
            cur_state += "Epoch: " + str(e) + " Iteration: " + str(batch_ndx) + " Total loss: " + str(total_loss) + "\n"
            cur_state += "BBox regression loss: " + str(box_regr_loss.item()) + "\n"
            cur_state += "Confidence (positive) loss: " + str(pos_conf_loss.item()) + "\n"
            cur_state += "Confidence (negative) loss: " + str(neg_conf_loss.item()) + "\n"
            cur_state += "Classification loss: " + str(probs_loss.item()) + "\n"
            cur_state += "Global step: " + str(global_step_cnt) + "\n"
            cur_state += "Step duration: " + str(iteration_duration) + " sec" + "\n"
            cur_state += "Batch size: " + str(train_batch_size) + "\n"
            cur_state += "Learning rate: " + str(scheduled_optimizer.scheduler.get_pars()[0])
            logger.log(cur_state)

            tboard_writer.add_scalar('Train/LossTotal', total_loss.item(), global_step_cnt)
            tboard_writer.add_scalar('Train/LossBBox', box_regr_loss.item(), global_step_cnt)
            tboard_writer.add_scalar('Train/LossConfP', pos_conf_loss.item(), global_step_cnt)
            tboard_writer.add_scalar('Train/LossConfN', neg_conf_loss.item(), global_step_cnt)
            tboard_writer.add_scalar('Train/LossClassif', probs_loss.item(), global_step_cnt)
            tboard_writer.add_scalar("Train/LearningRate", scheduled_optimizer.scheduler.get_pars()[0], global_step_cnt)

            if autosave_period_batches is not None:
                if batch_ndx % autosave_period_batches == 0:
                    if global_step_cnt > 0:
                        save_model(model, "Ep"+str(e)+"Btch"+str(batch_ndx))
                        logger.log("Model saved at epoch: " + str(e) + " iteration: " + str(batch_ndx))

            if valid_period:
                if batch_ndx % valid_period == 0:
                    if global_step_cnt > 0:
                        # show 1 batch
                        model.eval()

                        for sample in validation_dataloader:
                            image_batch = sample["input"]
                            target_batch = sample["target"]
                            if CUDA_IS_AVAILABLE:
                                image_batch = image_batch.to(device=processor_device)
                                target_batch = target_batch.to(device=processor_device)
                            prediction_batch = model.forward(image_batch)
                            if CUDA_IS_AVAILABLE:
                                prediction_batch = prediction_batch.cpu()
                                target_batch = target_batch.cpu()
                                image_batch = image_batch.cpu()

                            box_regr_loss, pos_conf_loss, neg_conf_loss, probs_loss = yolo_v1_loss(prediction_batch,
                                                                                                   target_batch)
                            val_loss = box_regr_loss + pos_conf_loss + neg_conf_loss + probs_loss
                            tboard_writer.add_scalar("Val/LossTotal", val_loss.item(), global_step_cnt)

                            pred_imgs = []
                            tgt_imgs = []
                            pred_by_imgs = []
                            target_by_imgs = []
                            for idx in range(prediction_batch.shape[0]):
                                det_objs = decode_output_tensor_act(prediction_batch[idx])
                                truth_objs = decode_output_tensor(target_batch[idx])
                                pred_img = get_prediction_img({"image": image_batch[idx], "objects": det_objs},
                                                              convert_rgb2bgr=False)
                                tgt_img = get_prediction_img({"image": image_batch[idx], "objects": truth_objs},
                                                             convert_rgb2bgr=False)
                                pred_img = img_yxc2cyx(pred_img)
                                tgt_img = img_yxc2cyx(tgt_img)
                                pred_imgs.append(torch.tensor(pred_img))
                                tgt_imgs.append(torch.tensor(tgt_img))
                                # Unchecked - mAP and AP
                                pred_by_imgs.append(det_objs)
                                target_by_imgs.append(truth_objs)

                            quality = mean_average_precision(pred_by_imgs, target_by_imgs)
                            for label, value in quality.items():
                                if label == "map":
                                    tboard_writer.add_scalar("Quality/mAP", value, global_step_cnt)
                                else:
                                    tboard_writer.add_scalar("Quality/AP_" + label, value, global_step_cnt)

                            img_grid_pred = torchvision.utils.make_grid(pred_imgs)
                            img_grid_tgt = torchvision.utils.make_grid(tgt_imgs)
                            tboard_writer.add_image('Valid/Predicted', img_tensor=img_grid_pred,
                                                    global_step=global_step_cnt, dataformats='CHW')
                            tboard_writer.add_image('Valid/Target', img_tensor=img_grid_tgt,
                                                    global_step=global_step_cnt, dataformats='CHW')
                            break

            global_step_cnt += 1

        if autosave_period_epochs is not None:
            if e % autosave_period_epochs == 0:
                save_model(model, "Ep"+str(e)+"Btch"+str(batch_ndx))
                logger.log("Model saved at epoch: " + str(e) + " iteration: " + str(batch_ndx))

    save_model(model, "FinalAfter" + str(epochs_cnt) + "Epochs")
    logger.log("Final model saved.")


main()
