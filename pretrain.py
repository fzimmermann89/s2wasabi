import gc
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import neptune
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="[\S\n\t\v ]*Triggered internally"
)
torch.set_num_threads(4)


from models import *
from util import load_wasabidata
from validation import val_FIXT2 as val
from warmup import WarmupLR
from wasabi2dDS_v5 import getLoader


# Set Parameters
LR = 3e-4
WD = 1e-3
FWL = 0.0
NET_NAME = "UNet"  # or MLP
N = 40
VALIDATIONFILENAME = "meas_MID22_20211005_WASABITI_sweep12_192px_fov256_4mm_TE3p42_TR7p08_FID82288_img-stack.nii"
datasetPath = Path("/scratch/zimmf/brainwebC/")
cachePath = Path("/lscratch/zimmf/")
outputPath = Path("/scratch/zimmf/Wasabi4/")

initial_lr = LR
weight_decay = WD
final_lr = 1e-5
l2 = FWL
l1 = l2 / 10
borderpad = 8
nettypes = {"UNet": unetonly3_lessFIXT2, "MLP": pixelwise}
print(f"{LR=} {WD=} {FWL=} {NET_NAME=}")
comment = f"{NET_NAME}-lr_{LR}-wd_{WD}"


if cachePath is not None:
    cachePath.mkdir(parents=True, exist_ok=True)
    Cache = TemporaryDirectory(dir=cachePath)
    cachePath = Path(Cache.name)
    c = shutil.copytree(datasetPath, cachePath, dirs_exist_ok=True)
    print(f"copied data to {c}")
else:
    cachePath = datasetPath
outputPath = outputPath / comment
outputPath.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime(f"%y%m%d_%H%M%S")
dl, dlVal, fw, size, offset, trec = getLoader(cachePath)
validationData = load_wasabidata(str(cachePath / VALIDATIONFILENAME), 0.0025)


# fmt: off
# Nromalization Constants
xm = torch.tensor([0.8, 10, 0, 3.75])[None, :, None, None]
xs = torch.tensor([0.5, 6, 0.3, 0.5])[None, :, None, None]
ym = torch.tensor(
    [0.15, 0.25, 0.3, 0.4, 0.5, 0.55, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.55, 0.5, 0.4, 0.3, 0.25, 0.15]
)[None, :, None, None]
ys = torch.tensor(
    [0.1, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
)[None, :, None, None]
# fmt: on


net = nettypes[NET_NAME](xm, xs, ym, ys)
print(
    f"{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters())]):.2e} parameters"
)
net = net.to("cuda")

neptune.init()  # credentials
logdir = outputPath / f"{timestamp}_log"
ex = neptune.create_experiment(
    comment,
    upload_source_files="*.py",
    params={
        "inital_lr": initial_lr,
        "final_lr": final_lr,
        "batchsize": dl.batch_size,
        "epochs": N,
        "fwloss_l": l2,
        "fwloss_lpre": l1,
        "number_params": sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, net.parameters())
            ]
        ),
        "weight_decay": weight_decay,
        "size": size,
        "borderpad": borderpad,
        "nofw": noFW,
        "model_name": NET_NAME,
    },
)
writer = SummaryWriter(log_dir=logdir)


optimizer = torch.optim.AdamW(
    net.parameters(), lr=initial_lr, weight_decay=weight_decay
)
scheduler = WarmupLR(
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, N - 2, eta_min=final_lr, verbose=True
    ),
    final_lr,
    2 * len(dl),
)
losses = []
net = net.to("cuda")
fw = fw.to("cuda")
criterion = torch.nn.MSELoss(reduction="none")
criterionG = torch.nn.GaussianNLLLoss(reduction="none")
running_loss1, running_loss1 = (
    torch.tensor(1, device="cuda"),
    torch.tensor(1, device="cuda"),
)
maskborder = F.pad(
    torch.ones(1, size[0] - 2 * borderpad, size[1] - 2 * borderpad),
    [borderpad] * 4,
    value=1e-4,
)
gc.collect()

for epoch in tqdm(range(N)):
    running_loss1, running_loss2 = (
        torch.zeros(1, device="cuda"),
        torch.zeros(1, device="cuda"),
    )
    net.train()
    for i, (xc, yc, ync, maskc, *_) in enumerate(dl):
        yn = ync.to("cuda", non_blocking=True)
        x = xc[:, (0, 2, 3)].to("cuda", non_blocking=True)
        y = yc.to("cuda", non_blocking=True)
        maskc *= maskborder
        maskc /= torch.clamp(torch.sum(maskc), min=1)
        mask = maskc.to("cuda", non_blocking=True)
        trueR2 = xc[:, 1].to("cuda", non_blocking=True)

        optimizer.zero_grad(True)

        if epoch < 2:
            xp = net(yn)
            lossG = torch.sum(criterionG(xp[:, :3], x, xp[:, 3:6]) * mask)
            lossM = torch.sum(criterion(xp[:, :3], x) * (mask))

            loss1 = 0.9 * lossG + 0.1 * lossM  # +0.01*lossF#+0.001*lossM
            scheduler.step()
            print("loss", lossG.item(), lossM.item(), scheduler.get_lr())
            del lossG, lossM
        else:
            xf = None
            xp = net(yn)
            loss1 = torch.sum(criterionG(xp[:, :3], x, xp[:, 3:6]) * mask)

        clamp = lambda x: torch.clamp(x, min=-1e3, max=1e3)
        yp = fw((clamp(xp[:, 0]), trueR2, clamp(xp[:, 1]), clamp(xp[:, 2])))

        if xp.shape[1] == 7:
            variance = xp[:, 6].unsqueeze(-1)
        elif xp.shape[1] == 6:
            variance = (yn - y).square().moveaxis(1, -1)
        else:
            variance = torch.ones_like(yp)
        loss2 = torch.sum(
            torch.mean(criterionG(y.moveaxis(1, -1), yp.moveaxis(1, -1), variance), -1)
            * mask.squeeze(1)
        )

        if epoch > 1:  # during warmup less FW Loss
            loss = loss1 + l2 * loss2
        else:
            loss = loss1 + l1 * loss2
        loss.backward()
        optimizer.step()

        running_loss1 += loss1.detach() / 3
        running_loss2 += loss2.detach()
        del x, y, yn, xp, yp, loss, loss1, loss2, mask, trueR2, variance
    scheduler.step()
    running_loss1 = running_loss1.item() / len(dl)
    running_loss2 = running_loss2.item() / len(dl)

    torch.cuda.synchronize()
    gc.collect()
    val(
        net,
        running_loss1,
        running_loss2,
        validationData,
        dlVal,
        optimizer,
        writer,
        ex,
        epoch,
        fw=fw,
        send_weights=False,
    )
    torch.cuda.synchronize()

print("Finished Training")
writer.close()
ex.stop()
gc.collect()
torch.cuda.memory.empty_cache()
net = net.cpu()

torch.save(net, outputPath / f"{timestamp}_{epoch}_net.pt")
torch.save(net.state_dict(), outputPath / f"{timestamp}_{epoch}_netstate.pt")
torch.save(optimizer.state_dict(), outputPath / f"{timestamp}_{epoch}_optimstate.pt")

print("done")
