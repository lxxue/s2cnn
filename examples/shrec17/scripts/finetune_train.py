# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn.functional as F
import torchvision

import os
import sys
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ModelNet, CacheNPY, ToMesh, ProjectOnSphere


def main(log_dir, model_path, ckpt_path, augmentation, dataset, num_cls, few, batch_size, num_workers, learning_rate):
    arguments = copy.deepcopy(locals())

    os.mkdir(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2(model_path, os.path.join(log_dir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    #model = mod.Model(55)
    model = mod.Model(num_cls)
    model.cuda()


    logger.info("==>Loading checkpoint ...")
    checkpoint = torch.load(ckpt_path)
    new_state_dict = model.state_dict()
    original_state_dict = checkpoint
    # check what is loaded and what is not
    #for k in original_state_dict:
    #    if k in new_state_dict:
    #        print("loading weight: {}".format(k))
    #    else:
    #        print("discard weight: {}".format(k))

    original_state_dict = {k: v for k, v in original_state_dict.items() if k in new_state_dict}
    new_state_dict.update(original_state_dict)
    model.load_state_dict(new_state_dict)
    

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    logger.info("{} paramerters in the last layer".format(sum(x.numel() for x in model.new_out_layer.parameters())))

    bw = model.bandwidths[0]

    # Load the dataset
    # Increasing `repeat` will generate more cached files
    train_transform = CacheNPY(prefix="b{}_".format(bw), repeat=augmentation, pick_randomly=True, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=True, random_translation=0.1),
            ProjectOnSphere(bandwidth=bw)
        ]
    ))

#    test_transform = torchvision.transforms.Compose([
#        CacheNPY(prefix="b64_", repeat=augmentation, pick_randomly=False, transform=torchvision.transforms.Compose(
#            [
#                ToMesh(random_rotations=True, random_translation=0.1),
#                ProjectOnSphere(bandwidth=64)
#            ]
#        )),
#        lambda xs: torch.stack([torch.FloatTensor(x) for x in xs])
#    ])
    test_transform=train_transform


    if "10" in dataset:
        train_data_type = "test"
        test_data_type = "train"
    else:
        train_data_type = "train"
        test_data_type = "test"

    train_set = ModelNet("/home/lixin/Documents/s2cnn/ModelNet", dataset, train_data_type, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    test_set = ModelNet("/home/lixin/Documents/s2cnn/ModelNet", dataset, test_data_type, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)

    def train_step(data, target):
        model.train()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    def test(epoch):
        predictions = []
        gt = []

        for batch_idx, (data, target) in enumerate(test_loader):
            model.eval()
            #batch_size, rep = data.size()[:2]
            #data = data.view(-1, *data.size()[2:])

            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                pred = model(data).data
            #pred = pred.view(batch_size*rep, -1)
            #pred = pred.sum(1)
        
            predictions.append(pred.cpu().numpy())
            #gt.append([target.cpu().numpy()]*rep)
            gt.append(target.cpu().numpy())

        predictions = np.concatenate(predictions)
        gt = np.concatenate(gt)

        predictions_class = np.argmax(predictions, axis=1)
        acc = np.sum(predictions_class == gt) / len(test_set)
        logger.info("Test Acc: {}".format(acc))
        return acc

    def get_learning_rate(epoch):
        limits = [100, 200]
        lrs = [1, 0.1, 0.01]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate

    best_acc = 0.
    for epoch in range(300):

        lr = get_learning_rate(epoch)
        logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, correct = train_step(data, target)

            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.3} <LOSS>={:.3} ACC={:.3} <ACC>={:.3} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / (batch_idx + 1),
                correct / len(data), total_correct / len(data) / (batch_idx + 1),
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

        test_acc = test(epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "best_state.pkl"))

        torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    #parser.add_argument("--dataset", choices={"test", "val", "train"}, default="train")
    parser.add_argument("--dataset", choices={"ModelNet30", "ModelNet10"}, default="train")
    parser.add_argument("--num_cls", type=int, choices={30, 10})
    parser.add_argument("--few", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.5)

    args = parser.parse_args()

    main(**args.__dict__)
