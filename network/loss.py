import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor


@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)

@ops.constexpr(reuse_result=True)
def get_pi(dtype=ms.float32):
    return Tensor(math.pi, dtype)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(box1)
    area2 = box_area(box2)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = ops.minimum(box1[:, None, 2:], box2[None, :, 2:]) - ops.maximum(box1[:, None, :2], box2[None, :, :2])
    inter = inter.clip(0, None)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter / (area1[:, None] + area2[None, :] - inter)  # iou = inter / (area1 + area2 - inter)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        x1, y1, w1, h1 = ops.split(box1, 1, 4)
        x2, y2, w2, h2 = ops.split(box2, 1, 4)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 1, 4)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 1, 4)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0, None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0, None)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1) # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / get_pi() ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox_iou_2(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    box1, box2 = box1.transpose(1, 0), box2.transpose(1, 0)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0, None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0, None)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:
                return iou # common IoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = "mean" # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = ops.Sigmoid()(pred) # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if mask is None:
            mask = ops.ones(loss.shape, loss.dtype)
        else:
            loss *= mask

        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class BCEWithLogitsLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.reduction = "mean" # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        loss = self.loss_fcn(pred, true)

        if mask is None:
            mask = ops.ones(loss.shape, loss.dtype)
        else:
            loss *= mask

        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False

        h = model.hyp  # hyperparameters
        self.hyp_anchor_t = h["anchor_t"]
        self.hyp_box = h['box']
        self.hyp_obj = h['obj']
        self.hyp_cls = h['cls']

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g),\
                             FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

        m = model.model[-1]  # Detect() module
        _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], dtype=ms.float32)

    def construct(self, p, targets):  # predictions, targets
        lcls = ops.zeros(1, ms.float32) # class loss
        lbox = ops.zeros(1, ms.float32) # box loss
        lobj = ops.zeros(1, ms.float32) # object loss

        tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask


        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            tobj = ops.zeros(pi.shape[:4], pi.dtype) # target obj

            n = b.shape[0]  # number of targets
            if n:
                _meta_pred = pi[b, a, gj, gi] #gather from (bs,na,h,w,nc)
                pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

                # Regression
                pxy = ops.Sigmoid()(pxy) * 2 - 0.5
                pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
                pbox = ops.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
                iou = iou * tmask
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = ops.Identity()(iou).clip(0, None)
                if self.sort_obj_iou:
                    _, j = ops.sort(iou)
                    b, a, gj, gi, iou, tmask_sorted = b[j], a[j], gj[j], gi[j], iou[j], tmask[j]
                else:
                    tmask_sorted = tmask
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou * tmask_sorted  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets

                    t[mnp.arange(n), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pcls, t, tmask[:, None])  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[layer_index]  # obj loss
            if self.autobalance:
                self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / obji.item()

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, ops.identity(ops.concat((lbox, lobj, lcls)))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
        gain = ops.ones(7, ms.int32) # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

        g = 0.5  # bias
        off = self._off * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape)[[3, 2, 3, 2]] # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
            # Matches
            # if nt:
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]
            j = ops.stack((ops.zeros_like(j), j, k, l, m)) # shape: (5, *)

            t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # t = t.repeat((5, 1, 1))[j]

            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
            offsets = offsets.view(-1, 2)
            # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]


            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
            anch += (anchors[a],)  # anchors
            tcls += (c,)  # class
            tmasks += (mask_m_t,)

        return ops.stack(tcls), \
               ops.stack(tbox), \
               ops.stack(indices), \
               ops.stack(anch), \
               ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask


class ComputeLossOTA(nn.Cell):
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        h = model.hyp
        self.hyp_box = h["box"]
        self.hyp_obj = h["obj"]
        self.hyp_cls = h["cls"]
        self.hyp_anchor_t = h["anchor_t"]

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

        m = model.model[-1]  # Detect() module
        _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance

        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.stride = m.stride

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], dtype=ms.float32)

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = ops.zeros(1, ms.float32), ops.zeros(1, ms.float32), ops.zeros(1, ms.float32)
        bs, as_, gjs, gis, targets, anchors, tmasks = self.build_targets(p, targets, imgs) # bs: (nl, bs*5*na*gt_max)

        pre_gen_gains = ()
        for pp in p:
            pre_gen_gains += (get_tensor(pp.shape)[[3, 2, 3, 2]],)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi, tmask = bs[i], as_[i], gjs[i], gis[i], tmasks[i]  # image, anchor, gridy, gridx, tmask
            tobj = ops.zeros_like(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            iou *= tmask
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.identity(iou).clip(0, None)) * tmask  # iou ratio

            # Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t, ops.tile(tmask[:, None], (1, t.shape[1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.identity(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.identity(ops.concat((lbox, lobj, lcls, loss)))


    def build_targets(self, p, targets, imgs):
        indices, anch, tmasks = self.find_3_positive(p, targets)

        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        matching_bs = ()
        matching_as = ()
        matching_gjs = ()
        matching_gis = ()
        matching_targets = ()
        matching_anchs = ()
        matching_tmasks = ()

        total_b = ()
        for i, _ in enumerate(p):
            total_b += (indices[i][0, :],)
        total_b = ops.stack(total_b, 0)
        _, total_b_indices = ops.sort(ops.cast(total_b, ms.float16))
        per_size_b = total_b.shape[1] // batch_size # [i*per_size_b:(i+1)*per_size_b]
        per_size_l = indices[0].shape[1]

        for batch_idx in range(p[0].shape[0]):
            this_target = targets[batch_idx, :, :]
            this_mask = this_target[:, 1] >= 0 # (1*gt_max,)

            txywh = this_target[:, 2:6] * img_size
            txyxy = xywh2xyxy(txywh)

            pxyxys = ()
            p_cls = ()
            p_obj = ()
            from_which_layer = ()
            all_b = ()
            all_a = ()
            all_gj = ()
            all_gi = ()
            all_anch = ()
            all_tmasks = ()

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i][0, :], indices[i][1, :], indices[i][2, :], indices[i][3, :]
                idx = total_b_indices[i, batch_idx * per_size_b:(batch_idx + 1) * per_size_b]
                # idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b += (b,)
                all_a += (a,)
                all_gj += (gj,)
                all_gi += (gi,)
                all_anch += (anch[i][idx],)
                all_tmasks += (tmasks[i][idx],)
                from_which_layer += (ops.ones(shape=(b.shape[0],), type=ms.int32) * i,)

                fg_pred = pi[b, a, gj, gi]
                p_obj += (fg_pred[:, 4:5],)
                p_cls += (fg_pred[:, 5:],)

                grid = ops.stack((gi, gj), axis=1)
                pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i] # / 8.
                pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * anch[i][idx] * self.stride[i]  # / 8.
                pxywh = ops.concat((pxy, pwh), axis=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys += (pxyxy,)

            pxyxys = ops.concat(pxyxys, axis=0) # nl * (5*na*gt_max, 4) -> cat -> (nl*5*na*gt_max, 4) # nt = bs * gt_max
            p_obj = ops.concat(p_obj, axis=0)
            p_cls = ops.concat(p_cls, axis=0)
            from_which_layer = ops.concat(from_which_layer, axis=0)
            all_b = ops.concat(all_b, axis=0)
            all_a = ops.concat(all_a, axis=0)
            all_gj = ops.concat(all_gj, axis=0)
            all_gi = ops.concat(all_gi, axis=0)
            all_anch = ops.concat(all_anch, axis=0)
            all_tmasks = ops.concat(all_tmasks, axis=0)

            pair_wise_iou = box_iou(txyxy, pxyxys) # (gt_max, nl*5*na*gt_max,)
            pair_wise_iou_loss = -ops.log(pair_wise_iou + 1e-8)

            v, _ = ops.sort(pair_wise_iou * all_tmasks[None, :] * this_mask[:, None])
            dynamic_ks = ops.cast(v[:, -10:].sum(1), ms.int32).clip(1, None)
            # v, _ = ops.top_k(pair_wise_iou * all_tmasks, 10, dim=-1)

            gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, 1], ms.int32),
                                           depth=self.nc,
                                           on_value=ops.ones(1, ms.int32),
                                           off_value=ops.zeros(1, ms.int32))
            gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, ms.float32), 1),
                                        (1, pxyxys.shape[0], 1))

            num_gt = this_target.shape[0]
            cls_preds_ = ops.Sigmoid()(ops.tile(ops.expand_dims(ops.cast(p_cls, ms.float32), 0), (num_gt, 1, 1))) * \
                         ops.Sigmoid()(ops.tile(ops.expand_dims(p_obj, 0) ,(num_gt, 1, 1)))

            y = ops.sqrt(cls_preds_)
            pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
                ops.log(y / (1 - y)),
                gt_cls_per_image,
                ops.ones(1, cls_preds_.dtype),
                ops.ones(1, cls_preds_.dtype),
                reduction="none",
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss # (gt_max, nl*5*na*gt_max)
            _cost_shape = cost.shape
            max_pos_cost = (cost * all_tmasks[None, :] * this_mask[:, None]).max()
            cost = ops.select(ops.cast(all_tmasks[None, :] * this_mask[:, None], ms.bool_),
                              cost,
                              ops.ones_like(cost) * (max_pos_cost + 1.))


            sort_cost, sort_idx = ops.sort(cost)
            pos_idx = ops.stack((mnp.arange(cost.shape[0]), dynamic_ks - 1), -1)
            pos_v = ops.gather_nd(sort_cost, pos_idx)
            matching_matrix = ops.cast(cost <= pos_v[:, None], ms.int32) * this_mask[:, None] * all_tmasks[None, :]

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = ops.argmin(cost, axis=0)
                anchor_matching_gt_mask_indices = ops.stack((cost_argmin, mnp.arange(cost_argmin.shape[0])), 1)
                # anchor_matching_gt_mask = ops.zeros_like(matching_matrix)
                # anchor_matching_gt_mask[anchor_matching_gt_mask_indices] = 1
                anchor_matching_gt_mask = ops.scatter_nd(anchor_matching_gt_mask_indices,
                                                         ops.ones_like(cost_argmin),
                                                         matching_matrix.shape)
                matching_matrix = matching_matrix * anchor_matching_gt_mask


            fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # (nl*5*na*gt_max,)
            all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32)
            matched_gt_inds = matching_matrix.argmax(0)
            # matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            this_target = this_target[matched_gt_inds]

            matching_bs += (all_b,)
            matching_as += (all_a,)
            matching_gjs += (all_gj,)
            matching_gis += (all_gi,)
            matching_targets += (this_target,)
            matching_anchs += (all_anch,)
            matching_tmasks += (all_tmasks,)

        # bs * (nl*5*na*gt_max,) -> (bs, nl*5*na*gt_max) -> (nl, bs*5*na*gt_max)
        matching_bs = ops.stack(matching_bs).view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_as = ops.stack(matching_as).view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_gjs = ops.stack(matching_gjs).view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_gis = ops.stack(matching_gis).view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_targets = ops.stack(matching_targets).view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6)
        matching_anchs = ops.stack(matching_anchs).view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2)
        matching_tmasks = ops.stack(matching_tmasks).view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks


    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, type=ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=ms.float32).view(na, 1), (1, nt)) # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 0.5  # bias
        off = self._off * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            # if nt:
            r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
            j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7) # (na*nt, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]
            j = ops.stack((ops.zeros_like(j), j, k, l, m))  # shape: (5, *)

            t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # t = t.repeat((5, 1, 1))[j]

            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets = offsets.view(-1, 2)
            # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)

            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks

if __name__ == '__main__':
    # python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml
    #   --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
    import yaml
    from pathlib import Path
    from mindspore import context
    from network.yolo import Model
    from config.args import get_args
    from utils.general import check_file, increment_path, colorstr

    opt = get_args()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    opt.total_batch_size = opt.batch_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    hyp['label_smoothing'] = opt.label_smoothing

    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    cfg = "./config/network_yolov7/yolov7.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    model.hyp = hyp
    model.set_train(True)
    compute_loss = ComputeLoss(model)

    x = Tensor(np.random.randn(2, 3, 160, 160), ms.float32)
    pred = model(x)
    print("pred: ", len(pred))
    # pred, grad = ops.value_and_grad(model, grad_position=0, weights=None)(x)
    # print("pred: ", len(pred), "grad: ", grad.shape)

    targets = Tensor(np.load("targets_bs2.npy"), ms.float32)
    # loss = compute_loss(pred, targets)
    (loss, _), grad = ops.value_and_grad(compute_loss, grad_position=0, weights=None, has_aux=True)(pred, targets)
    print("loss: ", loss)