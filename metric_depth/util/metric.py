import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    delta3 = torch.sum(thresh < 1.25).float() / len(thresh)
    delta4 = torch.sum(thresh < 1.20).float() / len(thresh)
    delta5 = torch.sum(thresh < 1.15).float() / len(thresh)
    delta6 = torch.sum(thresh < 1.10).float() / len(thresh)
    delta7 = torch.sum(thresh < 1.05).float() / len(thresh)

    diff = pred - target
    abs_rel = torch.mean(torch.abs(diff) / target)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))

    return {'delta1.25':delta3*100., 'delta1.20':delta4*100.,'delta1.15':delta5*100., 'delta1.10':delta6*100., 'delta1.05':delta7*100., 'mae':mae, 'absrel': abs_rel, 'rmse':rmse}


def booster_metrics(d, gt, valid):
    error = np.abs(d-gt)
    error[valid==0] = 0

    thresh = np.maximum((d[valid > 0] / gt[valid > 0]), (gt[valid > 0] / d[valid > 0]))
    delta3 = (thresh < 1.25).astype(np.float32).mean()
    delta4 = (thresh < 1.20).astype(np.float32).mean()
    delta5 = (thresh < 1.15).astype(np.float32).mean()
    delta6 = (thresh < 1.10).astype(np.float32).mean()
    delta7 = (thresh < 1.05).astype(np.float32).mean()

    avgerr = error[valid>0].mean()
    abs_rel = (error[valid>0]/gt[valid>0]).mean()

    rms = (d-gt)**2
    rms = np.sqrt( rms[valid>0].mean() )

    return {'delta1.25':delta3*100., 'delta1.20':delta4*100.,'delta1.15':delta5*100., 'delta1.10':delta6*100., 'delta1.05':delta7*100., 'mae':avgerr, 'absrel': abs_rel, 'rmse':rms, 'errormap':error*(valid>0)}
