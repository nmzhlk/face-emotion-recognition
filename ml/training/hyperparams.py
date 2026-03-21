import optuna
import pickle


CACHE_TO_RAM = False


import os
import time
import pandas
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import (
    MultiStepLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import WeightedRandomSampler


import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)


from utils.Custom_models import Resnet_Custom
from utils.Datasets import AffectNet_dataset

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

affectnet_labels_names = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


train_augmentations = torch.nn.Sequential(
    T.RandomResizedCrop((84, 70), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(7.0),
)


def train_epoch(model, optimizer, criteria, dataloader, epoch=None):
    global device
    model.train()

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device, non_blocking=True)
        if CACHE_TO_RAM:
            inputs = train_augmentations(inputs)
        outputs = model(inputs)
        loss = criteria(outputs, labels.to(device, non_blocking=True))

        loss.backward()
        optimizer.step()

        # metrics.update(outputs, labels, loss=loss.item())


def test_epoch(model, criteria, dataloader, epoch=None, to_show=False):
    global device
    model.eval()

    num = 0
    y_true = []
    y_pred = []
    loss_all = 0
    with torch.no_grad():
        for inputs, labels in dataloader:

            outputs = model(inputs.to(device, non_blocking=True))
            loss = criteria(outputs, labels.to(device, non_blocking=True))

            loss_all += loss
            preds = torch.argmax(outputs, dim=1)
            y_true.append(labels)
            y_pred.append(preds)

            # metrics.update(outputs=outputs, labels=labels, loss=loss.item(), is_test=True)

    print(f"Epo test  {epoch}:   Loss: {loss:0.3f}")

    from sklearn.metrics import classification_report

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    target_names = affectnet_labels_names

    print(
        classification_report(
            y_true, y_pred, target_names=target_names, zero_division=np.nan
        )
    )
    print("")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    if to_show:
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(
            xticks_rotation="vertical"
        )
        plt.show()

    return f1_score(y_true, y_pred, average="macro")


# def train_epoch(model, optimizer, criteria, dataloader, epoch=None):
#     global device

# def test_epoch(model, criteria, dataloader, epoch=None, to_show=False):
#     global device

#     return 2.1


def update_unfreezed_layers(epoch, model, optimizer):
    layer_configs = {
        3: ("layer1", 1e-4, 1e-5),
        5: ("layer2", 1e-4, 1e-5),
        7: ("layer3", 1e-4, 1e-5),
    }
    if epoch in layer_configs:
        layer_name, lr, wd = layer_configs[epoch]

        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = True

        add_param_group(optimizer, model, layer_name, lr, wd)


def add_param_group(optimizer, model, layer_name, lr=1e-4, wd=1e-5):

    existing_params = set()
    for pg in optimizer.param_groups:
        existing_params.update(pg["params"])

    new_params = [
        param
        for name, param in model.named_parameters()
        if (layer_name in name and param.requires_grad and param not in existing_params)
    ]
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": lr, "weight_decay": wd})


def save_model(model, epo="test", to_state=False):
    name = f"ML/models/Resnet_Custom_{str(epo)}.pth"
    if to_state:
        torch.save(model.state_dict(), name)
    else:
        torch.save(model, name)


# OOOPS https://arxiv.org/pdf/2105.03588


num_epo = 20
start_epo = 1
batch_size = 32

if CACHE_TO_RAM:
    transforms = T.Compose(
        [
            # T.Resize((84, 70)),\
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_transforms = T.Compose(
        [
            # T.Resize((84, 70)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

else:
    transforms = T.Compose(
        [
            T.RandomResizedCrop((84, 70), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(7),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_transforms = T.Compose(
        [
            T.CenterCrop((84, 70)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def optina_f1(trial):
    global study

    torch.cuda.empty_cache()
    model = Resnet_Custom(output_shape=len(affectnet_labels_names))
    model_name = "Resnet_AffectNet_optuna"
    model.to(device)

    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optim_choise = trial.suggest_categorical("optim_choise", ["AdamW", "SGD", ""])

    optimizer = torch.optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )

    # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40], gamma=0.4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)

    best_f1 = 0
    best_f1 = test_epoch(model, criteria, dataloader_test, 0)
    try:
        for epoch in range(start_epo, num_epo + 1):

            update_unfreezed_layers(epoch, model, optimizer)

            train_epoch(model, optimizer, criteria, dataloader_train, epoch)

            f1_macro = test_epoch(model, criteria, dataloader_test, epoch)

            scheduler.step()

            if f1_macro > best_f1:
                best_f1 = f1_macro

    except KeyboardInterrupt:
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)
        df = study.trials_dataframe()
        df.to_csv("study_inter.csv")
        pickle.dump(study, "experiments_inter.pkl")

    f1_macro = test_epoch(model, criteria, dataloader_test, epoch, to_show=True)
    if f1_macro > best_f1:
        best_f1 = f1_macro

    return best_f1


if __name__ == "__main__":

    dataset_train = AffectNet_dataset(transform=transforms, cache_to_ram=CACHE_TO_RAM)
    dataset_test = AffectNet_dataset(
        transform=test_transforms, is_test=True, cache_to_ram=CACHE_TO_RAM
    )

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if CACHE_TO_RAM else num_workers,  # RAM cache needs no workers
        pin_memory=True,
        persistent_workers=not CACHE_TO_RAM,
        prefetch_factor=2 if not CACHE_TO_RAM else None,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if CACHE_TO_RAM else num_workers,
        pin_memory=True,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(optina_f1, n_trials=60)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    df = study.trials_dataframe()
    df.to_csv("study.csv")
    pickle.dump(study, "experiments.pkl")
