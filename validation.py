from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gc
from torch import tensor
import torchvision
from functools import partial, reduce
from itertools import chain
import seaborn as sns


def removeNone(*args):
    return zip(*filter(lambda x: not any(el is None for el in x),zip(*args)))


def val_FIXT2(
    net, trainingloss1, trainingloss2, phantomdata, dlVal, optimizer, writer, ex, epoch, fw, training_lossD=None, name='', send_weights=False
):
    path = Path(writer.get_logdir()).absolute()
    torch.cuda.synchronize()
    if epoch % 4 == 0:
        for wname, weight in net.named_parameters():
            try:
                writer.add_histogram(wname, weight.cpu(), epoch)
                writer.add_histogram(f"{wname}.grad", weight.grad.cpu(), epoch)
                ex.log_metric(f"weights/norm/{wname}", epoch, weight.data.detach().cpu().norm())
                ex.log_metric(f"weights/gradnorm/{wname}", epoch, weight.grad.data.detach().cpu().norm())
            except:
                pass
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"lr_{i}", param_group["lr"], epoch)
            ex.log_metric(f"lr_{i}", epoch, param_group["lr"])
    with torch.no_grad():
        net.eval()
        gnllloss = tensor(0.0).cuda()
        fw_mseloss = tensor(0.0).cuda()
        mseloss = torch.zeros(3).cuda()
        accumvar = None#torch.zeros(4).cuda()
        criterionG = torch.nn.GaussianNLLLoss(reduction="none")

        nbins = 250
        hists = [np.zeros((nbins, nbins)) for i in range(6)]
        histbins = [None] * 6

        for nbatch, (x, y, yn, datamask, *_) in enumerate(dlVal):
            y = y.to("cuda", non_blocking=True)
            yn = yn.to("cuda", non_blocking=True)
            x = x.to("cuda", non_blocking=True)

            padmask = F.pad(datamask[..., 16:-16, 16:-16], [16, 16, 16, 16], value=0.0)
            masknorm = (padmask / torch.clamp(torch.sum(padmask), min=1)).to("cuda", non_blocking=True)
            maskcpu = datamask.squeeze_().numpy() > 0

            xp = net(yn)

            gnllloss += torch.sum(criterionG(xp[:, :3], x[:, (0, 2, 3)], xp[:, 3:6]) * masknorm) / 3
            err = F.mse_loss(xp[:, :3], x[:, (0, 2, 3)], reduction="none")
            mseloss += torch.sum(masknorm * err, (0, -1, -2))
            if accumvar is None:
                  accumvar=torch.zeros(xp.shape[1]-3).cuda()
            accumvar += torch.sum(masknorm * xp[:, 3:], (0, -1, -2))
            trueR2 = x[:, 1]
            clamp = lambda x: torch.clamp_(x, min=-1e3, max=1e3)
            yp = torch.nan_to_num_(fw((clamp(xp[:, 0]), trueR2, clamp(xp[:, 1]), clamp(xp[:, 2]))), nan=0, posinf=100, neginf=-100)
            fw_err = torch.mean(F.mse_loss(y, yp, reduction="none"), 1, keepdim=True)
            fw_mseloss += torch.sum(masknorm * fw_err)
        for i, (var, e) in enumerate(
            zip(xp[:, 3:].cpu().swapaxes(0, 1).numpy(), 
            chain(err.cpu().swapaxes(0, 1).numpy(), fw_err.cpu().swapaxes(0, 1).numpy()) if xp.shape[1]>6 else err.cpu().swapaxes(0, 1).numpy() )
        ):
            if histbins[i] is None:
                histbins[i] = [
                    np.logspace(np.log10(np.clip(w.min() / 30, 1e-7, 1e4)), np.log10(np.clip(np.nanpercentile(w, 99.9) * 30, 1e-5, 1e6)), nbins + 1)
                    for w in (var[maskcpu], e[maskcpu])
                ]
            hists[i] += np.histogram2d(var[maskcpu], e[maskcpu], bins=histbins[i])[0]
        gnllloss = gnllloss.item() / len(dlVal)
        fw_mseloss = fw_mseloss.item() / len(dlVal)
        mseloss = mseloss.cpu() / len(dlVal)
        accumvar = accumvar.cpu() / len(dlVal)

        print(epoch, trainingloss1, trainingloss2, mseloss, fw_mseloss, accumvar, gnllloss, training_lossD)

        xs = net.xs.squeeze().cpu()

        f, axs = plt.subplots(1, 5, tight_layout=True, figsize=(20, 3), dpi=100)
        try:
            hists,histbins=removeNone(hists,histbins)
            for hist, bins, ax in zip(hists, histbins, axs.ravel()):
                im = hist / np.sum(hist)
                X, Y = np.meshgrid(*[(b[1:] + b[:-1]) / 2 for b in bins])
                c = ax.pcolormesh(
                    X,
                    Y,
                    im.T,
                    cmap=sns.color_palette("rocket_r", as_cmap=True),
                    norm=matplotlib.colors.LogNorm(vmin=np.clip(np.min(im), 1e-6, 1e-2), vmax=np.clip(np.max(im), 1e-6, 1)),
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("var")
                ax.set_ylabel("err")
                plt.colorbar(c, ax=ax)
            ex.log_image("correlations/f2", epoch, f)
            plt.savefig(str(path / f'{name}_correlations_{epoch}.pdf'))
            plt.close()
        except Exception as e:
            print("error plotting var/err2. msg:", e)
        for i in range(len(mseloss)):
            writer.add_scalar(f"Loss/Val_mse/{i}_var", mseloss[i] / (xs[i]) ** 2, epoch)
            writer.add_scalar(f"Loss/Val_mse/{i}", mseloss[i], epoch)
        for i in range(len(accumvar)):
            writer.add_scalar(f"Var/{i}_var", accumvar[i], epoch)
        writer.add_scalar("Loss/Val", gnllloss, epoch)
        writer.add_scalar("Loss/Val_fw", fw_mseloss, epoch)
        if training_lossD is not None:
            writer.add_scalar("Loss/Train_D", training_lossD, epoch)
        writer.add_scalar("Loss/Train1", trainingloss1, epoch)
        writer.add_scalar("Loss/Train2", trainingloss2, epoch)

        for i in range(len(mseloss)):
            ex.log_metric(f"Loss/Val_mse/{i}_var", epoch, mseloss[i] / (xs[i]) ** 2)
            ex.log_metric(f"Loss/Val_mse/{i}", epoch, mseloss[i])
        for i in range(len(accumvar)):
            ex.log_metric(f"Var/{i}_var", epoch, accumvar[i])
        ex.log_metric("Loss/Val", epoch, gnllloss)
        ex.log_metric("Loss/Val_fw", epoch, fw_mseloss)
        if training_lossD is not None:
            ex.log_metric("Loss/Train_D", epoch, training_lossD)
        ex.log_metric("Loss/Train1", epoch, trainingloss1)
        ex.log_metric("Loss/Train2", epoch, trainingloss2)

        xpdata = xp[0].cpu()
        xpdata[:3][torch.broadcast_to(~torch.as_tensor(maskcpu[0]), xpdata[:3].shape)] = np.nan
        xpdata = xpdata[..., 16:-16, 16:-16]
        gt = x[0, (0, 2, 3)].cpu()[..., 16:-16, 16:-16]
        e = err[0].cpu()[..., 16:-16, 16:-16]
        xpdata[0] = 1 / xpdata[0]
        # xpdata[1] = 1 / xpdata[1]
        xpdata[2] = (xpdata[2]) / 3.75
        gt[0] = 1 / gt[0]
        # gt[1] = 1 / gt[1]
        gt[2] = (gt[2]) / 3.75

        ranges2 = [(0, 4), (-0.7, 0.7), (0.8, 1.2)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(xpdata[:3, ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred", f, epoch, close=False)
        ex.log_image("Val/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_pred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(2, 3, tight_layout=True, figsize=(16, 6), dpi=100)
        for i, ax in zip(np.array(xpdata[3:, ...]), axs[0].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        for i, ax in zip(np.array(e), axs[1].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred-Var", f, epoch, close=False)
        ex.log_image("Val/Var", epoch, f)
        plt.savefig(str(path / f'{name}_var_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(gt[(0, 1, 2), ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("GT", f, epoch, close=False)
        ex.log_image("Val/GT", epoch, f)
        plt.savefig(str(path / f'{name}_gt_{epoch}.pdf'))
        plt.close()

        x = x.cpu()
        xp = xp.cpu()

        def logimage(name, image, epoch):
            writer.add_image(name, image, epoch)
            ex.log_image(f"images/{name.replace('-','/')}", epoch, image.moveaxis(0, -1).numpy())

        phantom = torch.as_tensor(phantomdata[0], dtype=torch.float32)
        phantom[:, ~phantomdata[1]] = net.ym.cpu().squeeze()[:, None]
        xpdata = net(phantom.cuda()[None, ...]).cpu().squeeze()
        xpdata[0] = 1 / xpdata[0]
        xpdata[2] = (xpdata[2]) / 3.75
        ranges2 = [(0, 4), (-0.7, 0.7), (0.7, 1.3)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, r, ax in zip(np.array(xpdata[(0, 1, 2), 32:-32, 32:-32]), ranges2, axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom", f, epoch, close=False)
        ex.log_image("Phantom/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_phantompred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, ax in zip(np.array(xpdata[(0 + 3, 1 + 3, 2 + 3), 32:-32, 32:-32]), axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=np.nanpercentile(i[32:-32, 32:-32], 1), vmax=np.nanpercentile(i[32:-32, 32:-32], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom-Var", f, epoch, close=False)
        ex.log_image("Phantom/Var", epoch, f)
        plt.savefig(str(path / f'{name}_phantomvar_{epoch}.pdf'))
        plt.close()

        if epoch % 4 == 3:
            checkpointpath = Path(writer.get_logdir()).absolute() / f"weights_{epoch}.pt"
            torch.save(
                {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),}, checkpointpath,
            )
        try:
            if send_weights:
                ex.send_artifact(str(checkpointpath))
        except Exception as e:
            print('weight upload error')
        if epoch == 0:
            netpath = Path(writer.get_logdir()).absolute() / f"net_{epoch}.pt"
            torch.save(net, netpath)
            #ex.send_artifact(str(netpath))
            writer.add_graph(net, y)
        torch.cuda.synchronize()
        writer.flush()
        gc.collect()


def val_FIXT2_onlyFW(
    net, trainingloss1, trainingloss2, phantomdata, dlVal, optimizer, writer, ex, epoch, fw, training_lossD=None, name='', send_weights=False
):
    path = Path(writer.get_logdir()).absolute()
    torch.cuda.synchronize()
    if epoch % 4 == 0:
        for name, weight in net.named_parameters():
            try:
                writer.add_histogram(name, weight.cpu(), epoch)
                writer.add_histogram(f"{name}.grad", weight.grad.cpu(), epoch)
                ex.log_metric(f"weights/norm/{name}", epoch, weight.data.detach().cpu().norm())
                ex.log_metric(f"weights/gradnorm/{name}", epoch, weight.grad.data.detach().cpu().norm())
            except:
                pass
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"lr_{i}", param_group["lr"], epoch)
            ex.log_metric(f"lr_{i}", epoch, param_group["lr"])
    with torch.no_grad():
        net.eval()
        gnllloss = tensor(0.0).cuda()
        fw_mseloss = tensor(0.0).cuda()
        mseloss = torch.zeros(3).cuda()
        accumvar = None#torch.zeros(4).cuda()

        nbins = 250
        hists = [np.zeros((nbins, nbins)) for i in range(1)]
        histbins = [None] * 1

        for nbatch, (x, y, yn, datamask, *_) in enumerate(dlVal):
            y = y.to("cuda", non_blocking=True)
            yn = yn.to("cuda", non_blocking=True)
            x = x.to("cuda", non_blocking=True)

            padmask = F.pad(datamask[..., 16:-16, 16:-16], [16, 16, 16, 16], value=0.0)
            masknorm = (padmask / torch.clamp(torch.sum(padmask), min=1)).to("cuda", non_blocking=True)
            maskcpu = datamask.squeeze_().numpy() > 0

            xp = net(yn)

            err = F.mse_loss(xp[:, :3], x[:, (0, 2, 3)], reduction="none")
            mseloss += torch.sum(masknorm * err, (0, -1, -2))
            if accumvar is None:
                accumvar=torch.zeros(xp.shape[1]-3).cuda()
            accumvar += torch.sum(masknorm * xp[:, 3:], (0, -1, -2))
            trueR2 = x[:, 1]
            clamp = lambda x: torch.clamp_(x, min=-1e3, max=1e3)
            yp = torch.nan_to_num_(fw((clamp(xp[:, 0]), trueR2, clamp(xp[:, 1]), clamp(xp[:, 2]))), nan=0, posinf=100, neginf=-100)
            fw_err = torch.mean(F.mse_loss(y, yp, reduction="none"), 1, keepdim=True)
            fw_mseloss += torch.sum(masknorm * fw_err)
        for i, (var, e) in enumerate(zip(xp[:, 3:].cpu().swapaxes(0, 1).numpy(), fw_err.cpu().swapaxes(0, 1).numpy())):
            if histbins[i] is None:
                histbins[i] = [
                    np.logspace(np.log10(np.clip(w.min() / 30, 1e-7, 1e4)), np.log10(np.clip(np.nanpercentile(w, 99.9) * 30, 1e-5, 1e6)), nbins + 1)
                    for w in (var[maskcpu], e[maskcpu])
                ]
            hists[i] += np.histogram2d(var[maskcpu], e[maskcpu], bins=histbins[i])[0]
        fw_mseloss = fw_mseloss.item() / len(dlVal)
        mseloss = mseloss.cpu() / len(dlVal)
        accumvar = accumvar.cpu() / len(dlVal)

        print(epoch, trainingloss1, trainingloss2, mseloss, fw_mseloss, accumvar)

        xs = net.xs.squeeze().cpu()

        f, axs = plt.subplots(1, 5, tight_layout=True, figsize=(20, 3), dpi=100)
        try:
            for hist, bins, ax in zip(hists, histbins, axs.ravel()):
                im = hist / np.sum(hist)
                X, Y = np.meshgrid(*[(b[1:] + b[:-1]) / 2 for b in bins])
                c = ax.pcolormesh(
                    X,
                    Y,
                    im.T,
                    cmap=sns.color_palette("rocket_r", as_cmap=True),
                    norm=matplotlib.colors.LogNorm(vmin=np.clip(np.min(im), 1e-6, 1e-2), vmax=np.clip(np.max(im), 1e-6, 1)),
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("var")
                ax.set_ylabel("err")
                plt.colorbar(c, ax=ax)
            ex.log_image("correlations/f2", epoch, f)
            plt.savefig(str(path / f'{name}_correlations_{epoch}.pdf'))
            plt.close()
        except Exception as e:
            print("error plotting var/err2. msg:", e)
        for i in range(len(mseloss)):
            writer.add_scalar(f"Loss/Val_mse/{i}_var", mseloss[i] / (xs[i]) ** 2, epoch)
            writer.add_scalar(f"Loss/Val_mse/{i}", mseloss[i], epoch)
        for i in range(len(accumvar)):
            writer.add_scalar(f"Var/{i}_var", accumvar[i], epoch)
        writer.add_scalar("Loss/Val", gnllloss, epoch)
        writer.add_scalar("Loss/Val_fw", fw_mseloss, epoch)
        if training_lossD is not None:
            writer.add_scalar("Loss/Train_D", training_lossD, epoch)
        writer.add_scalar("Loss/Train1", trainingloss1, epoch)
        writer.add_scalar("Loss/Train2", trainingloss2, epoch)

        for i in range(len(mseloss)):
            ex.log_metric(f"Loss/Val_mse/{i}_var", epoch, mseloss[i] / (xs[i]) ** 2)
            ex.log_metric(f"Loss/Val_mse/{i}", epoch, mseloss[i])
        for i in range(len(accumvar)):
            ex.log_metric(f"Var/{i}_var", epoch, accumvar[i])
        ex.log_metric("Loss/Val", epoch, gnllloss)
        ex.log_metric("Loss/Val_fw", epoch, fw_mseloss)
        if training_lossD is not None:
            ex.log_metric("Loss/Train_D", epoch, training_lossD)
        ex.log_metric("Loss/Train1", epoch, trainingloss1)
        ex.log_metric("Loss/Train2", epoch, trainingloss2)

        xpdata = xp[0].cpu()
        xpdata[:3][torch.broadcast_to(~torch.as_tensor(maskcpu[0]), xpdata[:3].shape)] = np.nan
        xpdata = xpdata[..., 16:-16, 16:-16]
        gt = x[0, (0, 2, 3)].cpu()[..., 16:-16, 16:-16]
        e = err[0].cpu()[..., 16:-16, 16:-16]
        xpdata[0] = 1 / xpdata[0]
        # xpdata[1] = 1 / xpdata[1]
        xpdata[2] = (xpdata[2]) / 3.75
        gt[0] = 1 / gt[0]
        # gt[1] = 1 / gt[1]
        gt[2] = (gt[2]) / 3.75

        ranges2 = [(0, 4), (-0.7, 0.7), (0.8, 1.2)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(xpdata[:3, ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred", f, epoch, close=False)
        ex.log_image("Val/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_pred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(2, 3, tight_layout=True, figsize=(16, 6), dpi=100)
        for i, ax in zip(np.array(xpdata[3:, ...]), axs[0].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        for i, ax in zip(np.array(e), axs[1].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred-Var", f, epoch, close=False)
        ex.log_image("Val/Var", epoch, f)
        plt.savefig(str(path / f'{name}_var_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(gt[(0, 1, 2), ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("GT", f, epoch, close=False)
        ex.log_image("Val/GT", epoch, f)
        plt.savefig(str(path / f'{name}_gt_{epoch}.pdf'))
        plt.close()

        x = x.cpu()
        xp = xp.cpu()

        def logimage(name, image, epoch):
            writer.add_image(name, image, epoch)
            ex.log_image(f"images/{name.replace('-','/')}", epoch, image.moveaxis(0, -1).numpy())

        phantom = torch.as_tensor(phantomdata[0], dtype=torch.float32)
        phantom[:, ~phantomdata[1]] = net.ym.cpu().squeeze()[:, None]
        xpdata = net(phantom.cuda()[None, ...]).cpu().squeeze()
        xpdata[0] = 1 / xpdata[0]
        xpdata[2] = (xpdata[2]) / 3.75
        ranges2 = [(0, 4), (-0.7, 0.7), (0.7, 1.3)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, r, ax in zip(np.array(xpdata[(0, 1, 2), 32:-32, 32:-32]), ranges2, axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom", f, epoch, close=False)
        ex.log_image("Phantom/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_phantompred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, ax in zip(np.array(xpdata[3:, 32:-32, 32:-32]), axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=np.nanpercentile(i[32:-32, 32:-32], 1), vmax=np.nanpercentile(i[32:-32, 32:-32], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom-Var", f, epoch, close=False)
        ex.log_image("Phantom/Var", epoch, f)
        plt.savefig(str(path / f'{name}_phantomvar_{epoch}.pdf'))
        plt.close()

        if epoch % 4 == 3:
            checkpointpath = Path(writer.get_logdir()).absolute() / f"weights_{epoch}.pt"
            torch.save(
                {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),}, checkpointpath,
            )
        try:
            if send_weights:
                ex.send_artifact(str(checkpointpath))
        except Exception as e:
            print('weight upload error')
        if epoch == 0:
            netpath = Path(writer.get_logdir()).absolute() / f"net_{epoch}.pt"
            torch.save(net, netpath)
            ex.send_artifact(str(netpath))
            writer.add_graph(net, y)
        torch.cuda.synchronize()
        writer.flush()
        gc.collect()


def val_FIXT2_onlyFWMSE(
    net, trainingloss1, trainingloss2, phantomdata, dlVal, optimizer, writer, ex, epoch, fw, training_lossD=None, name='', send_weights=False
):
    path = Path(writer.get_logdir()).absolute()
    torch.cuda.synchronize()
    if epoch % 4 == 0:
        for name, weight in net.named_parameters():
            try:
                writer.add_histogram(name, weight.cpu(), epoch)
                writer.add_histogram(f"{name}.grad", weight.grad.cpu(), epoch)
                ex.log_metric(f"weights/norm/{name}", epoch, weight.data.detach().cpu().norm())
                ex.log_metric(f"weights/gradnorm/{name}", epoch, weight.grad.data.detach().cpu().norm())
            except:
                pass
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"lr_{i}", param_group["lr"], epoch)
            ex.log_metric(f"lr_{i}", epoch, param_group["lr"])
    with torch.no_grad():
        net.eval()
        gnllloss = tensor(0.0).cuda()
        fw_mseloss = tensor(0.0).cuda()
        mseloss = torch.zeros(3).cuda()
        accumvar = torch.zeros(0).cuda()

        nbins = 250
        hists = []  # [np.zeros((nbins, nbins)) for i in range(1)]
        histbins = []  # [None] * 1

        for nbatch, (x, y, yn, datamask, *_) in enumerate(dlVal):
            y = y.to("cuda", non_blocking=True)
            yn = yn.to("cuda", non_blocking=True)
            x = x.to("cuda", non_blocking=True)

            padmask = F.pad(datamask[..., 16:-16, 16:-16], [16, 16, 16, 16], value=0.0)
            masknorm = (padmask / torch.clamp(torch.sum(padmask), min=1)).to("cuda", non_blocking=True)
            maskcpu = datamask.squeeze_().numpy() > 0

            xp = net(yn)

            err = F.mse_loss(xp[:, :3], x[:, (0, 2, 3)], reduction="none")
            mseloss += torch.sum(masknorm * err, (0, -1, -2))
            # accumvar += torch.sum(masknorm * xp[:, 3:], (0, -1, -2))
            trueR2 = x[:, 1]
            clamp = lambda x: torch.clamp_(x, min=-1e3, max=1e3)
            yp = torch.nan_to_num_(fw((clamp(xp[:, 0]), trueR2, clamp(xp[:, 1]), clamp(xp[:, 2]))), nan=0, posinf=100, neginf=-100)
            fw_err = torch.mean(F.mse_loss(y, yp, reduction="none"), 1, keepdim=True)
            fw_mseloss += torch.sum(masknorm * fw_err)
        #         for i, (var, e) in enumerate(zip(xp[:, 3:].cpu().swapaxes(0, 1).numpy(), fw_err.cpu().swapaxes(0, 1).numpy())):
        #             if histbins[i] is None:
        #                 histbins[i] = [np.logspace(np.log10(np.clip(w.min() / 30, 1e-7, 1e4)), np.log10(np.clip(np.nanpercentile(w, 99.9) * 30, 1e-5, 1e6)), nbins + 1) for w in (var[maskcpu], e[maskcpu])]
        #             hists[i] += np.histogram2d(var[maskcpu], e[maskcpu], bins=histbins[i])[0]
        fw_mseloss = fw_mseloss.item() / len(dlVal)
        mseloss = mseloss.cpu() / len(dlVal)
        # accumvar = accumvar.cpu() / len(dlVal)

        print(epoch, trainingloss1, trainingloss2, mseloss, fw_mseloss)

        xs = net.xs.squeeze().cpu()

        f, axs = plt.subplots(1, 5, tight_layout=True, figsize=(20, 3), dpi=100)
        #         try:
        #             for hist, bins, ax in zip(hists, histbins, axs.ravel()):
        #                 im = hist / np.sum(hist)
        #                 X, Y = np.meshgrid(*[(b[1:] + b[:-1]) / 2 for b in bins])
        #                 c = ax.pcolormesh(X, Y, im.T, cmap=sns.color_palette("rocket_r", as_cmap=True), norm=matplotlib.colors.LogNorm(vmin=np.clip(np.min(im), 1e-6, 1e-2), vmax=np.clip(np.max(im), 1e-6, 1)))
        #                 ax.set_xscale("log")
        #                 ax.set_yscale("log")
        #                 ax.set_xlabel("var")
        #                 ax.set_ylabel("err")
        #                 plt.colorbar(c, ax=ax)
        #             ex.log_image("correlations/f2", epoch, f)
        #             plt.savefig(str(path/f'{name}_correlations_{epoch}.pdf'))
        #             plt.close()
        #         except Exception as e:
        #             print("error plotting var/err2. msg:", e)
        for i in range(len(mseloss)):
            writer.add_scalar(f"Loss/Val_mse/{i}_var", mseloss[i] / (xs[i]) ** 2, epoch)
            writer.add_scalar(f"Loss/Val_mse/{i}", mseloss[i], epoch)
        for i in range(len(accumvar)):
            writer.add_scalar(f"Var/{i}_var", accumvar[i], epoch)
        writer.add_scalar("Loss/Val", gnllloss, epoch)
        writer.add_scalar("Loss/Val_fw", fw_mseloss, epoch)
        if training_lossD is not None:
            writer.add_scalar("Loss/Train_D", training_lossD, epoch)
        writer.add_scalar("Loss/Train1", trainingloss1, epoch)
        writer.add_scalar("Loss/Train2", trainingloss2, epoch)

        for i in range(len(mseloss)):
            ex.log_metric(f"Loss/Val_mse/{i}_var", epoch, mseloss[i] / (xs[i]) ** 2)
            ex.log_metric(f"Loss/Val_mse/{i}", epoch, mseloss[i])
        for i in range(len(accumvar)):
            ex.log_metric(f"Var/{i}_var", epoch, accumvar[i])
        ex.log_metric("Loss/Val", epoch, gnllloss)
        ex.log_metric("Loss/Val_fw", epoch, fw_mseloss)
        if training_lossD is not None:
            ex.log_metric("Loss/Train_D", epoch, training_lossD)
        ex.log_metric("Loss/Train1", epoch, trainingloss1)
        ex.log_metric("Loss/Train2", epoch, trainingloss2)

        xpdata = xp[0].cpu()
        xpdata[:3][torch.broadcast_to(~torch.as_tensor(maskcpu[0]), xpdata[:3].shape)] = np.nan
        xpdata = xpdata[..., 16:-16, 16:-16]
        gt = x[0, (0, 2, 3)].cpu()[..., 16:-16, 16:-16]
        e = err[0].cpu()[..., 16:-16, 16:-16]
        xpdata[0] = 1 / xpdata[0]
        # xpdata[1] = 1 / xpdata[1]
        xpdata[2] = (xpdata[2]) / 3.75
        gt[0] = 1 / gt[0]
        # gt[1] = 1 / gt[1]
        gt[2] = (gt[2]) / 3.75

        ranges2 = [(0, 4), (-0.7, 0.7), (0.8, 1.2)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(xpdata[:3, ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred", f, epoch, close=False)
        ex.log_image("Val/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_pred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(2, 3, tight_layout=True, figsize=(16, 6), dpi=100)
        for i, ax in zip(np.array(xpdata[3:, ...]), axs[0].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        for i, ax in zip(np.array(e), axs[1].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred-Var", f, epoch, close=False)
        ex.log_image("Val/Var", epoch, f)
        plt.savefig(str(path / f'{name}_var_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=100)
        for i, r, ax in zip(np.array(gt[(0, 1, 2), ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("GT", f, epoch, close=False)
        ex.log_image("Val/GT", epoch, f)
        plt.savefig(str(path / f'{name}_gt_{epoch}.pdf'))
        plt.close()

        x = x.cpu()
        xp = xp.cpu()

        def logimage(name, image, epoch):
            writer.add_image(name, image, epoch)
            ex.log_image(f"images/{name.replace('-','/')}", epoch, image.moveaxis(0, -1).numpy())

        phantom = torch.as_tensor(phantomdata[0], dtype=torch.float32)
        phantom[:, ~phantomdata[1]] = net.ym.cpu().squeeze()[:, None]
        xpdata = net(phantom.cuda()[None, ...]).cpu().squeeze()
        xpdata[0] = 1 / xpdata[0]
        xpdata[2] = (xpdata[2]) / 3.75
        ranges2 = [(0, 4), (-0.7, 0.7), (0.7, 1.3)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, r, ax in zip(np.array(xpdata[(0, 1, 2), 32:-32, 32:-32]), ranges2, axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom", f, epoch, close=False)
        ex.log_image("Phantom/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_phantompred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, ax in zip(np.array(xpdata[3:, 32:-32, 32:-32]), axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=np.nanpercentile(i[32:-32, 32:-32], 1), vmax=np.nanpercentile(i[32:-32, 32:-32], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom-Var", f, epoch, close=False)
        ex.log_image("Phantom/Var", epoch, f)
        plt.savefig(str(path / f'{name}_phantomvar_{epoch}.pdf'))
        plt.close()

        if epoch % 10 == 9:
            checkpointpath = Path(writer.get_logdir()).absolute() / f"weights_{epoch}.pt"
            torch.save(
                {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),}, checkpointpath,
            )
        try:
            if send_weights:
                ex.send_artifact(str(checkpointpath))
        except Exception as e:
            print('weight upload error')
        if epoch == 0:
            netpath = Path(writer.get_logdir()).absolute() / f"net_{epoch}.pt"
            torch.save(net, netpath)
            try:
                if send_weights:
                    ex.send_artifact(str(netpath))
            except Exception as e:
                print('weight upload error')
            writer.add_graph(net, y)
        torch.cuda.synchronize()
        writer.flush()
        gc.collect()


def val_noFW(
    net, trainingloss1, trainingloss2, phantomdata, dlVal, optimizer, writer, ex, epoch, fw=None, training_lossD=None, name='', send_weights=False
):
    path = Path(writer.get_logdir()).absolute()
    torch.cuda.synchronize()
    if epoch % 2 == 0:
        for name, weight in net.named_parameters():
            try:
                writer.add_histogram(name, weight.cpu(), epoch)
                writer.add_histogram(f"{name}.grad", weight.grad.cpu(), epoch)
                ex.log_metric(f"weights/norm/{name}", epoch, weight.data.detach().cpu().norm())
                ex.log_metric(f"weights/gradnorm/{name}", epoch, weight.grad.data.detach().cpu().norm())
            except:
                pass
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"lr_{i}", param_group["lr"], epoch)
            ex.log_metric(f"lr_{i}", epoch, param_group["lr"])
    with torch.no_grad():
        net.eval()
        gnllloss = tensor(0.0).cuda()
        fw_mseloss = tensor(0.0).cuda()
        mseloss = torch.zeros(3).cuda()
        accumvar = None#torch.zeros(3).cuda()
        criterionG = torch.nn.GaussianNLLLoss(reduction="none")

        nbins = 250
        hists = [np.zeros((nbins, nbins)) for i in range(3)]
        histbins = [None] * 3

        for nbatch, (x, y, yn, datamask, *_) in enumerate(dlVal):
            y = y.to("cuda", non_blocking=True)
            yn = yn.to("cuda", non_blocking=True)
            x = x.to("cuda", non_blocking=True)

            padmask = F.pad(datamask[..., 16:-16, 16:-16], [16, 16, 16, 16], value=0.0)
            masknorm = (padmask / torch.clamp(torch.sum(padmask), min=1)).to("cuda", non_blocking=True)
            maskcpu = datamask.squeeze_().numpy() > 0

            xp = net(yn)

            gnllloss += torch.sum(criterionG(xp[:, :3], x[:, (0, 2, 3)], xp[:, 3:6]) * masknorm) / 4
            err = F.mse_loss(xp[:, :3], x[:, (0, 2, 3)], reduction="none")
            mseloss += torch.sum(masknorm * err, (0, -1, -2))
            if accumvar is None:
                accumvar=torch.zeros(xp.shape[1]-3).cuda()
            accumvar += torch.sum(masknorm * xp[:, 3:], (0, -1, -2))

            # yp = torch.nan_to_num_(fw(torch.clamp_(xp.swapaxes(0, 1)[:4], min=-1e3, max=1e3)), nan=0, posinf=100, neginf=-100)
            # fw_err = torch.mean(F.mse_loss(y, yp, reduction="none"), 1, keepdim=True)
            # fw_mseloss += torch.sum(masknorm * fw_err)
        for i, (var, e) in enumerate(zip(xp[:, 3:].cpu().swapaxes(0, 1).numpy(), chain(err.cpu().swapaxes(0, 1).numpy()))):
            if histbins[i] is None:
                histbins[i] = [
                    np.logspace(np.log10(np.clip(w.min() / 30, 1e-7, 1e4)), np.log10(np.clip(np.nanpercentile(w, 99.9) * 30, 1e-5, 1e6)), nbins + 1)
                    for w in (var[maskcpu], e[maskcpu])
                ]
            hists[i] += np.histogram2d(var[maskcpu], e[maskcpu], bins=histbins[i])[0]
        gnllloss = gnllloss.item() / len(dlVal)
        # fw_mseloss = fw_mseloss.item() / len(dlVal)
        mseloss = mseloss.cpu() / len(dlVal)
        accumvar = accumvar.cpu() / len(dlVal)

        print(trainingloss1, trainingloss2, mseloss, accumvar, gnllloss, training_lossD)

        xs = net.xs.squeeze().cpu()

        f, axs = plt.subplots(1, 5, tight_layout=True, figsize=(20, 3), dpi=150)
        try:
            for hist, bins, ax in zip(hists, histbins, axs.ravel()):
                im = hist / np.sum(hist)
                X, Y = np.meshgrid(*[(b[1:] + b[:-1]) / 2 for b in bins])
                c = ax.pcolormesh(
                    X,
                    Y,
                    im.T,
                    cmap=sns.color_palette("rocket_r", as_cmap=True),
                    norm=matplotlib.colors.LogNorm(vmin=np.clip(np.min(im), 1e-6, 1e-2), vmax=np.clip(np.max(im), 1e-6, 1)),
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("var")
                ax.set_ylabel("err")
                plt.colorbar(c, ax=ax)
            ex.log_image("correlations/f2", epoch, f)
            plt.savefig(str(path / f'{name}_correlations_{epoch}.pdf'))
            plt.close()
        except Exception as e:
            print("error plotting var/err2. msg:", e)
        for i in range(len(mseloss)):
            writer.add_scalar(f"Loss/Val_mse/{i}_var", mseloss[i] / (xs[i]) ** 2, epoch)
            writer.add_scalar(f"Loss/Val_mse/{i}", mseloss[i], epoch)
        for i in range(len(accumvar)):
            writer.add_scalar(f"Var/{i}_var", accumvar[i], epoch)
        writer.add_scalar("Loss/Val", gnllloss, epoch)
        writer.add_scalar("Loss/Val_fw", fw_mseloss, epoch)
        if training_lossD is not None:
            writer.add_scalar("Loss/Train_D", training_lossD, epoch)
        writer.add_scalar("Loss/Train1", trainingloss1, epoch)
        writer.add_scalar("Loss/Train2", trainingloss2, epoch)

        for i in range(len(mseloss)):
            ex.log_metric(f"Loss/Val_mse/{i}_var", epoch, mseloss[i] / (xs[i]) ** 2)
            ex.log_metric(f"Loss/Val_mse/{i}", epoch, mseloss[i])
        for i in range(len(accumvar)):
            ex.log_metric(f"Var/{i}_var", epoch, accumvar[i])
        ex.log_metric("Loss/Val", epoch, gnllloss)
        # ex.log_metric("Loss/Val_fw", epoch, fw_mseloss)
        if training_lossD is not None:
            ex.log_metric("Loss/Train_D", epoch, training_lossD)
        ex.log_metric("Loss/Train1", epoch, trainingloss1)
        ex.log_metric("Loss/Train2", epoch, trainingloss2)

        xpdata = xp[0].cpu()
        xpdata[:3][torch.broadcast_to(~torch.as_tensor(maskcpu[0]), xpdata[:3].shape)] = np.nan
        xpdata = xpdata[..., 16:-16, 16:-16]
        gt = x[0, (0, 2, 3)].cpu()[..., 16:-16, 16:-16]
        e = err[0].cpu()[..., 16:-16, 16:-16]
        xpdata[0] = 1 / xpdata[0]
        # xpdata[1] = 1 / xpdata[1]
        xpdata[2] = (xpdata[2]) / 3.75
        gt[0] = 1 / gt[0]
        # gt[1] = 1 / gt[1]
        gt[2] = (gt[2]) / 3.75

        ranges2 = [(0, 4), (-0.7, 0.7), (0.8, 1.2)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=150)
        for i, r, ax in zip(np.array(xpdata[:3, ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred", f, epoch, close=False)
        ex.log_image("Val/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_pred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(2, 3, tight_layout=True, figsize=(16, 6), dpi=150)
        for i, ax in zip(np.array(xpdata[3:, ...]), axs[0].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        for i, ax in zip(np.array(e), axs[1].ravel()):
            c = ax.matshow(i, vmin=np.nanpercentile(i[16:-16, 16:-16], 1), vmax=np.nanpercentile(i[16:-16, 16:-16], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Pred-Var", f, epoch, close=False)
        ex.log_image("Val/Var", epoch, f)
        plt.savefig(str(path / f'{name}_var_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 3), dpi=150)
        for i, r, ax in zip(np.array(gt[(0, 1, 2), ...]), ranges2, axs.ravel()):
            c = ax.matshow(i, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("GT", f, epoch, close=False)
        ex.log_image("Val/GT", epoch, f)
        plt.savefig(str(path / f'{name}_gt_{epoch}.pdf'))
        plt.close()

        x = x.cpu()
        xp = xp.cpu()

        def logimage(name, image, epoch):
            writer.add_image(name, image, epoch)
            ex.log_image(f"images/{name.replace('-','/')}", epoch, image.moveaxis(0, -1).numpy())

        phantom = torch.as_tensor(phantomdata[0], dtype=torch.float32)
        phantom[:, ~phantomdata[1]] = net.ym.cpu().squeeze()[:, None]
        xpdata = net(phantom.cuda()[None, ...]).cpu().squeeze()
        # xpdata[torch.broadcast_to(~torch.as_tensor(phantomdata[1]), xpdata.shape)] = np.nan
        xpdata[0] = 1 / xpdata[0]
        xpdata[2] = (xpdata[2]) / 3.75
        ranges2 = [(0, 4), (-0.7, 0.7), (0.7, 1.3)]

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, r, ax in zip(np.array(xpdata[(0, 1, 2), 32:-32, 32:-32]), ranges2, axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=r[0], vmax=r[1])
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom", f, epoch, close=False)
        ex.log_image("Phantom/Pred", epoch, f)
        plt.savefig(str(path / f'{name}_phantompred_{epoch}.pdf'))
        plt.close()

        f, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3), dpi=150)
        for i, ax in zip(np.array(xpdata[(0 + 3, 1 + 3, 2 + 3), 32:-32, 32:-32]), axs.ravel()):
            i[~phantomdata[1][32:-32, 32:-32]] = np.nan
            c = ax.matshow(i.T, vmin=np.nanpercentile(i[32:-32, 32:-32], 1), vmax=np.nanpercentile(i[32:-32, 32:-32], 99))
            plt.colorbar(c, ax=ax)
        writer.add_figure("Phantom-Var", f, epoch, close=False)
        ex.log_image("Phantom/Var", epoch, f)
        plt.savefig(str(path / f'{name}_phantomvar_{epoch}.pdf'))
        plt.close()

        if epoch % 4 == 2:
            checkpointpath = Path(writer.get_logdir()).absolute() / f"weights_{epoch}.pt"
            torch.save(
                {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),}, checkpointpath,
            )
        try:
            if send_weights:
                ex.send_artifact(str(checkpointpath))
        except Exception as e:
            print('weight upload error')
        if epoch == 0:
            netpath = Path(writer.get_logdir()).absolute() / f"net_{epoch}.pt"
            torch.save(net, netpath)
            ex.send_artifact(str(netpath))
            writer.add_graph(net, y)
            # torch.onnx.export(net, y, str(Path(writer.get_logdir()).absolute() / "model.onnx"), verbose=False, input_names=['input'], output_names=['output'],opset_version=13,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        torch.cuda.synchronize()
        writer.flush()
        gc.collect()
