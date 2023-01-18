import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor

__all__ = ['ComputeLoss',
           'ComputeLossOTA',
           'ComputeLossAuxOTA',
           'ComputeLossOTA_dynamic']

CLIP_VALUE = 1000.
EPS = 1e-7
PI = Tensor(math.pi, ms.float32)

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

def batch_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[:, :, 0] = x[:, :, 0] - x[:, :, 2] / 2  # top left x
    y[:, :, 1] = x[:, :, 1] - x[:, :, 3] / 2  # top left y
    y[:, :, 2] = x[:, :, 0] + x[:, :, 2] / 2  # bottom right x
    y[:, :, 3] = x[:, :, 1] + x[:, :, 3] / 2  # bottom right y
    return y

def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def batch_box_area(box):
    return (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])

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

    expand_size_1 = box2.shape[0]
    expand_size_2 = box1.shape[0]

    box1 = ops.tile(ops.expand_dims(box1, 1), (1, expand_size_1, 1))
    box2 = ops.tile(ops.expand_dims(box2, 0), (expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # inter = ops.minimum(box1[:, None, 2:], box2[None, :, 2:]) - ops.maximum(box1[:, None, :2], box2[None, :, :2])
    inter = ops.minimum(box1[..., 2:], box2[..., 2:]) - ops.maximum(box1[..., :2], box2[..., :2])
    inter = inter.clip(0., None)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # zhy_test
    return inter / (area1[:, None] + area2[None, :] - inter).clip(EPS, None)  # iou = inter / (area1 + area2 - inter)

def batch_box_iou(batch_box1, batch_box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[B, N, 4])
        box2 (Tensor[B, M, 4])
    Returns:
        iou (Tensor[B, N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = batch_box_area(batch_box1)
    area2 = batch_box_area(batch_box2)

    expand_size_1 = batch_box2.shape[1]
    expand_size_2 = batch_box1.shape[1]
    batch_box1 = ops.tile(ops.expand_dims(batch_box1, 2), (1, 1, expand_size_1, 1))
    batch_box2 = ops.tile(ops.expand_dims(batch_box2, 1), (1, expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = ops.minimum(batch_box1[..., 2:], batch_box2[..., 2:]) - \
            ops.maximum(batch_box1[..., :2], batch_box2[..., :2])
    inter = inter.clip(0., None)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]
    # zhy_test
    return inter / (area1[:, :, None] + area2[:, None, :] - inter).clip(EPS, None)  # iou = inter / (area1 + area2 - inter)

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
        # w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        # w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1) # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # v = (4 / get_pi(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                v = (4 / PI.astype(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox_iou_2(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # box1/2, (n, 4) -> (4, n)
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
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

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
    def __init__(self, bce_weight=None, bce_pos_weight=None, gamma=1.5, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = ops.Sigmoid()(pred) # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)

class BCEWithLogitsLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)

class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False

        h = model.opt # hyperparameters
        self.hyp_anchor_t = h.anchor_t
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g),\
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

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
        lcls, lbox, lobj = 0., 0., 0.

        tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
        tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
                                               ops.stop_gradient(indices), ops.stop_gradient(anchors), \
                                               ops.stop_gradient(tmasks)

        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
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
                # iou = iou * tmask
                # lbox += (1.0 - iou).mean()  # iou loss
                lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None)  # iou loss

                # Objectness
                iou = ops.stop_gradient(iou).clip(0, None)
                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * iou) * tmask  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets
                    t[mnp.arange(n), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[layer_index]  # obj loss
            if self.autobalance:
                self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / ops.stop_gradient(obji).item()

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls

        return loss * bs, ops.stop_gradient(ops.stack((lbox, lobj, lcls, loss)))

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
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

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
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1)) #.astype(ms.int32)
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1)) #.astype(ms.int32)
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # # 1. Original
            # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
            # offsets = offsets.view(-1, 2)

            # 2. Faster,
            tag1, tag2 = ops.tile(j[:, None], (1, 2)), ops.tile(k[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
            offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy - offsets, ms.int32)
            gij = gij[:]
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
        h = model.opt
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls
        self.hyp_anchor_t = h.anchor_t

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

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
        lcls, lbox, lobj = 0., 0., 0.
        bs, as_, gjs, gis, targets, anchors, tmasks = self.build_targets(p, targets, imgs) # bs: (nl, bs*5*na*gt_max)
        bs, as_, gjs, gis, targets, anchors, tmasks = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                                      ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                                      ops.stop_gradient(targets), ops.stop_gradient(anchors), \
                                                      ops.stop_gradient(tmasks)

        pre_gen_gains = ()
        for pp in p:
            pre_gen_gains += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)

        # Losses
        # for i, pi in enumerate(p):  # layer index, layer predictions
        for i in range(self.nl): # layer index
            pi = p[i] # layer predictions
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
            # iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            iou = bbox_iou(pbox, selected_tbox, xywh=True, CIoU=True).view(-1)
            lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None) # iou loss

            # Objectness
            tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) * tmask  # iou ratio

            # Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t, ops.tile(tmask[:, None], (1, t.shape[1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets, imgs):
        indices, anch, tmasks = self.find_3_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 3 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(3 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(3 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 3 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 3 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1) # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1) # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1) # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None] # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        v, _ = ops.top_k(pair_wise_iou, 10) # (bs, gt_max, 10)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 10), ms.int32) # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1), (1, n_gt_max, 1, 1)) # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1) # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 10, sorted=True) # (bs, gt_max, 10)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 10), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2, 1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32) # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1) # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None], (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        # zhy_test
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        # zhy_test
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

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
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.))
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            # j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            # offsets = offsets.view(-1, 2) # (5*na*nt, 2)
            # # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # faster,
            tag1, tag2 = ops.tile(j[:, None], (1, 2)), ops.tile(k[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, ...], offsets[3, ...])
            offsets_new[2, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, ...], offsets[4, ...])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks

class ComputeLossAuxOTA(nn.Cell):
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA, self).__init__()
        h = model.opt
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls
        self.hyp_anchor_t = h.anchor_t

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

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
        lcls, lbox, lobj = 0., 0., 0.
        targets_ori = targets
        bs, as_, gjs, gis, targets, anchors, tmasks = self.build_targets(p[:self.nl], targets_ori, imgs) # bs: (nl, bs*3*na*gt_max)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux, tmasks_aux = self.build_targets_2(p[:self.nl], targets_ori, imgs) # bs: (nl, bs*5*na*gt_max)

        bs, as_, gjs, gis, targets, anchors, tmasks = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                                      ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                                      ops.stop_gradient(targets), ops.stop_gradient(anchors), \
                                                      ops.stop_gradient(tmasks)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux, tmasks_aux = ops.stop_gradient(bs_aux),\
                                                                                  ops.stop_gradient(as_aux_), \
                                                                                  ops.stop_gradient(gjs_aux),\
                                                                                  ops.stop_gradient(gis_aux), \
                                                                                  ops.stop_gradient(targets_aux), \
                                                                                  ops.stop_gradient(anchors_aux), \
                                                                                  ops.stop_gradient(tmasks_aux)

        pre_gen_gains = ()
        # pre_gen_gains_aux = ()
        for pp in p[:self.nl]:
            pre_gen_gains += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)
            # pre_gen_gains_aux += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)

        # Losses
        for i in range(self.nl): # layer index
            pi = p[i] # layer predictions
            pi_aux = p[i + self.nl]
            b, a, gj, gi, tmask = bs[i], as_[i], gjs[i], gis[i], tmasks[i]  # image, anchor, gridy, gridx, tmask
            b_aux, a_aux, gj_aux, gi_aux, tmask_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i], tmasks_aux[i]
            tobj = ops.zeros_like(pi[..., 0])  # target obj
            tobj_aux = ops.zeros_like(pi_aux[..., 0])  # target obj


            # 1. Branch1, Compute main branch loss
            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # 1.1. Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            # iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            iou = bbox_iou(pbox, selected_tbox, xywh=True, CIoU=True).view(-1)
            lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None) # iou loss
            # 1.2. Objectness
            tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) * tmask  # iou ratio
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)
            # 1.3. Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t, ops.tile(tmask[:, None], (1, t.shape[1])))  # BCE

            # 2. Branch2, Compute Aux branch loss
            n_aux = b_aux.shape[0]  # number of targets
            ps_aux = pi[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
            # 2.1. Regression
            grid_aux = ops.stack([gi_aux, gj_aux], axis=1)
            pxy_aux = ops.Sigmoid()(ps_aux[:, :2]) * 2. - 0.5
            pwh_aux = (ops.Sigmoid()(ps_aux[:, 2:4]) * 2) ** 2 * anchors_aux[i]
            pbox_aux = ops.concat((pxy_aux, pwh_aux), 1)  # predicted box
            selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox_aux[:, :2] -= grid_aux
            # iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            iou_aux = bbox_iou(pbox_aux, selected_tbox_aux, xywh=True, CIoU=True).view(-1)
            lbox += 0.25 * ((1.0 - iou_aux) * tmask_aux).sum() / tmask_aux.astype(iou_aux.dtype).sum().clip(1, None)  # iou loss
            # 1.2. Objectness
            tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou_aux).clip(0, None)) * tmask_aux  # iou ratio
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += 0.25 * obji_aux * self.balance[i]  # obj loss
            # 1.3. Classification
            selected_tcls_aux = ops.cast(targets_aux[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t_aux = ops.ones_like(ps_aux[:, 5:]) * self.cn  # targets
                t_aux[mnp.arange(n_aux), selected_tcls_aux] = self.cp
                lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux, ops.tile(tmask_aux[:, None], (1, t_aux.shape[1])))  # BCE

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets, imgs):
        indices, anch, tmasks = self.find_3_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 3 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(3 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(3 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 3 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 3 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1) # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1) # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1) # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None] # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        # Top 20 iou sum for aux, default 10
        v, _ = ops.top_k(pair_wise_iou, 20) # (bs, gt_max, 20)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 20), ms.int32) # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1), (1, n_gt_max, 1, 1)) # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1) # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 20, sorted=True) # (bs, gt_max, 20)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 20), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2, 1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32) # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1) # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None], (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def build_targets_2(self, p, targets, imgs):
        indices, anch, tmasks = self.find_5_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 5 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(5 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(5 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 5 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 5 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1)  # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1)  # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1)  # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None]  # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        # Top 20 iou sum for aux, default 10
        v, _ = ops.top_k(pair_wise_iou, 20)  # (bs, gt_max, 20)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 20), ms.int32)  # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1),
                              (1, n_gt_max, 1, 1))  # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1)  # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 20, sorted=True)  # (bs, gt_max, 20)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 20), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2,
                                                                                           1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(
            1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32)  # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1)  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None],
                                   (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * \
                           matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

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
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.))
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            # j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            # offsets = offsets.view(-1, 2) # (5*na*nt, 2)
            # # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # faster,
            tag1, tag2 = ops.tile(j[:, None], (1, 2)), ops.tile(k[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, ...], offsets[3, ...])
            offsets_new[2, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, ...], offsets[4, ...])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)  # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0  # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt))  # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 1.0  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain  # (na, nt, 7)
            # Matches
            r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
            j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1) # filter
            t = t.view(-1, 7)  # (na*nt, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.)).astype(ms.int32)
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.)).astype(ms.int32)
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets = offsets.view(-1, 2) # (5*na*nt, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)

            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks

class ComputeLossOTA_dynamic(nn.Cell):
    # run with mindspore version 2.0.0
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA_dynamic, self).__init__()
        h = model.opt
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls
        self.hyp_anchor_t = h.anchor_t

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

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
        lcls, lbox, lobj = 0., 0., 0.
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs) # bs: (nl, bs*5*na*gt_max)
        bs, as_, gjs, gis, targets, anchors = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                              ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                              ops.stop_gradient(targets), ops.stop_gradient(anchors)
        pre_gen_gains = ()
        for pp in p:
            pre_gen_gains += (get_tensor(pp.shape, pp.dtype)[[3, 2, 3, 2]],)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx, tmask
            tobj = ops.zeros_like(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]

            # selected_tbox[:, 0:2] -= grid
            # _selected_tbox_1, _selected_tbox_2 = ops.split(selected_tbox, 1, 2) # ms 1.8.1
            _selected_tbox_1, _selected_tbox_2 = ops.split(selected_tbox, 2, 1) # ms 2.0.0
            _selected_tbox_1 -= grid
            selected_tbox = ops.concat((_selected_tbox_1, _selected_tbox_2), 1)

            iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean() # iou loss

            # Objectness
            tobj[b, a, gj, gi] = ops.ones(iou.shape, iou.dtype) * \
                                 ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) # iou ratio

            # Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls.view(n)] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets, imgs):
        indices, anch = self.find_3_positive(p, targets) # 3 * (4, nm_2), 3 * (nm_2, 2)
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]
        targets = targets.view(-1, 6)  # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0  # (bs*gt_max,)
        targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6)  # (nt, 6)

        matching_bs = ()
        matching_as = ()
        matching_gjs = ()
        matching_gis = ()
        matching_targets = ()
        matching_anchs = ()
        matching_from_which_layers = ()

        for batch_idx in range(p[0].shape[0]):
            b_idx = (targets[:, 0] == batch_idx) # (n_tb,)
            # this_target = ops.masked_select(targets, ops.tile(b_idx[:, None], (1, 6))).view(-1, 6) # (n_tb, 6)
            this_target = ops.masked_select(targets, ops.broadcast_to(b_idx[:, None], b_idx.shape + (6,))).view(-1, 6)  # (n_tb, 6)
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1] # (n_tb, 4)
            txyxy = self.xywh2xyxy(txywh)
            pxyxys = ()
            p_cls = ()
            p_obj = ()
            from_which_layer = ()
            all_b = ()
            all_a = ()
            all_gj = ()
            all_gi = ()
            all_anch = ()

            for i, pi in enumerate(p):
                # b, a, gj, gi = indices[i]
                # b, a, gj, gi = ops.split(indices[i], 0, 4) # ms 1.8.1
                b, a, gj, gi = ops.split(indices[i], 1, 0)  # ms 2.0.0
                b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)

                idx = (b == batch_idx) # (n_tp_b,)
                b, a, gj, gi = ops.masked_select(b, idx), ops.masked_select(a, idx), \
                               ops.masked_select(gj, idx), ops.masked_select(gi, idx)
                # _this_anch = ops.masked_select(anch[i], ops.tile(idx[:, None], (1, 2))).view(-1, 2) # (n_tp_b, 2)
                _this_anch = ops.masked_select(anch[i], ops.broadcast_to(idx[:, None], idx.shape + (2,))).view(-1, 2)  # (n_tp_b, 2)
                all_b += (b,)
                all_a += (a,)
                all_gj += (gj,)
                all_gi += (gi,)
                all_anch += (_this_anch,)
                from_which_layer += (ops.ones((b.shape[0],), ms.int32) * i,)

                fg_pred = pi[b, a, gj, gi] # (n_tp_b, 85)
                p_obj += (fg_pred[:, 4:5],)
                p_cls += (fg_pred[:, 5:],)

                grid = ops.stack((gi, gj), axis=1) # (n_tp_b, 2)
                pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
                pxywh = ops.concat((pxy, pwh), axis=-1) # (n_tp_b, 4)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys += (pxyxy,)

            pxyxys = ops.concat(pxyxys, axis=0) # (n_tp, 4)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = ops.concat(p_obj, axis=0) # (n_tp, 1)
            p_cls = ops.concat(p_cls, axis=0) # (n_tp, 80)
            from_which_layer = ops.concat(from_which_layer, axis=0)
            all_b = ops.concat(all_b, axis=0) # (n_tp,)
            all_a = ops.concat(all_a, axis=0)
            all_gj = ops.concat(all_gj, axis=0)
            all_gi = ops.concat(all_gi, axis=0)
            all_anch = ops.concat(all_anch, axis=0) # (n_tp, 2)

            pair_wise_iou = box_iou(txyxy, pxyxys) # (n_tb, 4), (n_tp, 4) -> (n_tb, n_tp)
            pair_wise_iou_loss = -ops.log(pair_wise_iou + 1e-8)

            v, _ = ops.top_k(pair_wise_iou, min(10, pair_wise_iou.shape[1])) # (n_tb, 10)
            dynamic_ks = v.sum(1).astype(ms.int32).clip(1, None) # (n_tb,)

            gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, 1], ms.int32),
                                           depth=self.nc,
                                           on_value=ops.ones(1, pair_wise_iou.dtype),
                                           off_value=ops.zeros(1, pair_wise_iou.dtype)) # (n_tb, 80)
            # gt_cls_per_image = ops.tile(gt_cls_per_image[:, None, :], (1, pxyxys.shape[0], 1)) # (n_tb, n_tp, 85)
            gt_cls_per_image = ops.broadcast_to(gt_cls_per_image[:, None, :], (gt_cls_per_image.shape[0], pxyxys.shape[0], gt_cls_per_image.shape[1]))  # (n_tb, n_tp, 85)

            num_gt = this_target.shape[0]
            # cls_preds_ = ops.Sigmoid()(ops.tile(p_cls[None, :, :], (num_gt, 1, 1))) * \
            #              ops.Sigmoid()(ops.tile(p_obj[None, :, :], (num_gt, 1, 1))) # (n_tb, n_tp, 80)
            cls_preds_ = ops.Sigmoid()(ops.broadcast_to(p_cls[None, :, :], (num_gt,) + p_cls.shape)) * \
                         ops.Sigmoid()(ops.broadcast_to(p_obj[None, :, :], (num_gt,) + p_obj.shape))  # (n_tb, n_tp, 80)

            y = ops.sqrt(cls_preds_)
            pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
                ops.log(y / (1 - y)),
                gt_cls_per_image,
                ops.ones(1, cls_preds_.dtype),
                ops.ones(1, cls_preds_.dtype),
                reduction="none",
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss  # (n_tb, n_tp)

            # 1. dynamic-k match with pynative and dynamic-shape
            matching_matrix = ops.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = ops.top_k(
                    cost[gt_idx], k=int(dynamic_ks[gt_idx].asnumpy())
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            # 2. dynamic-k match with pynative and static-shape
            # sort_cost, sort_idx = ops.top_k(-cost, 10, sorted=True)
            # sort_cost = -sort_cost
            # pos_idx = ops.stack((mnp.arange(cost.shape[0]), dynamic_ks - 1), -1)
            # pos_v = ops.gather_nd(sort_cost, pos_idx)
            # matching_matrix = ops.cast(cost <= pos_v[:, None], ms.int32)

            anchor_matching_gt = matching_matrix.sum(0)  # (n_tp,)
            anchor_matching_mask = anchor_matching_gt > 1  # (n_tp,)
            anchor_matching_mask_idx = ops.masked_select(mnp.arange(cost.shape[1]), anchor_matching_mask)
            if anchor_matching_mask.astype(cost.dtype).sum() > 0:
                # cost_argmin = ops.argmin(ops.masked_select(cost,
                #                                            ops.tile(anchor_matching_mask[None, :], (cost.shape[0], 1))
                #                                            ).view(cost.shape[0], -1), axis=0)
                cost_argmin = ops.argmin(ops.masked_select(cost,
                                                           ops.broadcast_to(anchor_matching_mask[None, :], (cost.shape[0],) + anchor_matching_mask.shape)
                                                           ).view(cost.shape[0], -1), axis=0)
                # matching_matrix[:, anchor_matching_mask_idx] *= 0.0
                matching_matrix *= 1 - anchor_matching_mask.astype(matching_matrix.dtype)
                matching_matrix[cost_argmin, anchor_matching_mask_idx] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # (n_tp,)

            if fg_mask_inboxes.astype(cost.dtype).sum() > 0.0:
                fg_mask_inboxes_idx = ops.masked_select(mnp.arange(cost.shape[1]), fg_mask_inboxes) # (n_tp_m,)

                # matched_gt_inds = matching_matrix[:, fg_mask_inboxes_idx].argmax(0) # (n_tb, n_tp) -> (n_tb, n_tp_m,) -> (n_tp_m,)
                matched_gt_inds = matching_matrix.transpose(1, 0)[fg_mask_inboxes_idx].argmax(1) # (n_tb, n_tp) -> (n_tp, n_tb) -> (n_tp_m, n_tb) -> (n_tp_m,)

                from_which_layer = from_which_layer[fg_mask_inboxes_idx] # (n_tp,) -> (n_tp_m,)
                all_b = all_b[fg_mask_inboxes_idx]
                all_a = all_a[fg_mask_inboxes_idx]
                all_gj = all_gj[fg_mask_inboxes_idx]
                all_gi = all_gi[fg_mask_inboxes_idx]
                all_anch = all_anch[fg_mask_inboxes_idx]

                this_target = this_target[matched_gt_inds] # (n_tb, 6) -> (n_tp_m, 6)

                matching_from_which_layers += (from_which_layer,)
                matching_bs += (all_b,)
                matching_as += (all_a,)
                matching_gjs += (all_gj,)
                matching_gis += (all_gi,)
                matching_targets += (this_target,)
                matching_anchs += (all_anch,)

        matching_bs = ops.concat(matching_bs, 0)
        matching_as = ops.concat(matching_as, 0)
        matching_gjs = ops.concat(matching_gjs, 0)
        matching_gis = ops.concat(matching_gis, 0)
        matching_targets = ops.concat(matching_targets, 0)
        matching_anchs = ops.concat(matching_anchs, 0)
        matching_from_which_layers = ops.concat(matching_from_which_layers, 0)

        _matching_bs = ()
        _matching_as = ()
        _matching_gjs = ()
        _matching_gis = ()
        _matching_targets = ()
        _matching_anchs = ()
        _matching_from_which_layers = ()

        for i in range(nl):
            layer_mask = matching_from_which_layers == i
            if layer_mask.astype(ms.float16).sum() > 0.0:
                layer_idx = ops.masked_select(mnp.arange(matching_bs.shape[0]), layer_mask)
                _matching_bs += (ops.stop_gradient(matching_bs[layer_idx]),)
                _matching_as += (ops.stop_gradient(matching_as[layer_idx]),)
                _matching_gjs += (ops.stop_gradient(matching_gjs[layer_idx]),)
                _matching_gis += (ops.stop_gradient(matching_gis[layer_idx]),)
                _matching_targets += (ops.stop_gradient(matching_targets[layer_idx]),)
                _matching_anchs += (ops.stop_gradient(matching_anchs[layer_idx]),)
            else:
                _matching_bs += (Tensor([], matching_bs.dtype),)
                _matching_as += (Tensor([], matching_as.dtype),)
                _matching_gjs += (Tensor([], matching_gjs.dtype),)
                _matching_gis += (Tensor([], matching_gis.dtype),)
                _matching_targets += (Tensor([], matching_targets.dtype),)
                _matching_anchs += (Tensor([], matching_anchs.dtype),)

        return _matching_bs, _matching_as, _matching_gjs, _matching_gis, _matching_targets, _matching_anchs

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = ops.zeros(x.shape, x.dtype)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)

        # targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6) # (nt, 6)
        targets = ops.masked_select(targets, ops.broadcast_to(mask_t[:, None], mask_t.shape + (6,))).view(-1, 6)  # (nt, 6)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain

        # ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        ai = ops.broadcast_to(mnp.arange(na, dtype=targets.dtype).view(na, 1), (na, nt))

        # (na, nt, 7)
        # targets = ops.concat((ops.tile(targets[None, :, :], (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # not support dynamic shape
        targets = ops.concat((ops.broadcast_to(targets[None, :, :], (na,) + targets.shape[:]), ai[:, :, None]), 2) # append anchor indices

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            if nt:
                r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio # (na, nt, 2)
                j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7) # (nm, 7)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm, 7)

                # t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy # (nm, 2)
                gxi = gain[[2, 3]] - gxy  # inverse
                jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
                lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
                j, k = jk[:, 0], jk[:, 1] # (nm,)
                l, m = lm[:, 0], lm[:, 1]

                # original
                j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, nm)

                # t = ops.tile(t, (5, 1, 1))  # shape(5, nm, 7)
                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7)  # (nm_2, 7)
                t = ops.broadcast_to(t, (5,) + t.shape)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm_2, 7)

                offsets = ops.zeros_like(gxy)[None, :, :] + off[:, None, :] # (5, nm, 2)
                # offsets = ops.masked_select(offsets, ops.tile(j[:, :, None], (1, 1, 2))).view(-1, 2)
                offsets = ops.masked_select(offsets, ops.broadcast_to(j[:, :, None], j.shape + (2,))).view(-1, 2)
            else:
                t = targets[0]
                offsets = 0
                print("warning: find_3_pos: this batch has no target!")

            # Define # (nm_2,)
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

        return indices, anch

class ComputeLossAuxOTA_dynamic(nn.Cell):
    # run with mindspore version 2.0.0
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA_dynamic, self).__init__()
        h = model.opt
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls
        self.hyp_anchor_t = h.anchor_t

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

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
        lcls, lbox, lobj = 0., 0., 0.
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs) # bs: (nl, bs*5*na*gt_max)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets, imgs)

        bs, as_, gjs, gis, targets, anchors = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                              ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                              ops.stop_gradient(targets), ops.stop_gradient(anchors)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = ops.stop_gradient(bs_aux),\
                                                                      ops.stop_gradient(as_aux_), \
                                                                      ops.stop_gradient(gjs_aux), \
                                                                      ops.stop_gradient(gis_aux), \
                                                                      ops.stop_gradient(targets_aux), \
                                                                      ops.stop_gradient(anchors_aux)
        pre_gen_gains = ()
        pre_gen_gains_aux = ()
        for pp in p[:self.nl]:
            pre_gen_gains += (get_tensor(pp.shape, pp.dtype)[[3, 2, 3, 2]],)
            pre_gen_gains_aux += (get_tensor(pp.shape, pp.dtype)[[3, 2, 3, 2]],)

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i + self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx, tmask
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]
            tobj = ops.zeros_like(pi[..., 0])  # target obj
            tobj_aux = ops.zeros_like(pi_aux[..., 0])

            # 1. Branch Common
            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            # selected_tbox[:, 0:2] -= grid
            # _selected_tbox_1, _selected_tbox_2 = ops.split(selected_tbox, 1, 2) # ms 1.8.1
            _selected_tbox_1, _selected_tbox_2 = ops.split(selected_tbox, 2, 1) # ms 2.0.0
            _selected_tbox_1 -= grid
            selected_tbox = ops.concat((_selected_tbox_1, _selected_tbox_2), 1)
            iou = bbox_iou_2(pbox, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean() # iou loss
            # Objectness
            tobj[b, a, gj, gi] = ops.ones(iou.shape, iou.dtype) * \
                                 ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) # iou ratio
            # Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls.view(n)] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            # 2. Branch Aux
            n_aux = b_aux.shape[0]  # number of targets
            ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
            # Regression
            grid_aux = ops.stack([gi_aux, gj_aux], axis=1)
            pxy_aux = ops.Sigmoid()(ps_aux[:, :2]) * 2. - 0.5
            pwh_aux = (ops.Sigmoid()(ps_aux[:, 2:4]) * 2) ** 2 * anchors_aux[i]
            pbox_aux = ops.concat((pxy_aux, pwh_aux), 1)  # predicted box
            selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
            _selected_tbox_1_aux, _selected_tbox_2_aux = ops.split(selected_tbox_aux, 2, 1)  # ms 2.0.0
            _selected_tbox_1_aux -= grid_aux
            selected_tbox_aux = ops.concat((_selected_tbox_1_aux, _selected_tbox_2_aux), 1)
            iou_aux = bbox_iou_2(pbox_aux, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss
            # Objectness
            tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = ops.ones(iou_aux.shape, iou_aux.dtype) * \
                                 ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou_aux).clip(0, None))  # iou ratio
            # Classification
            selected_tcls_aux = ops.cast(targets_aux[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t_aux = ops.ones_like(ps_aux[:, 5:]) * self.cn  # targets
                t_aux[mnp.arange(n_aux), selected_tcls_aux.view(n_aux)] = self.cp
                lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE


            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets, imgs):
        indices, anch = self.find_3_positive(p, targets) # 3 * (4, nm_2), 3 * (nm_2, 2)
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]
        targets = targets.view(-1, 6)  # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0  # (bs*gt_max,)
        targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6)  # (nt, 6)

        matching_bs = ()
        matching_as = ()
        matching_gjs = ()
        matching_gis = ()
        matching_targets = ()
        matching_anchs = ()
        matching_from_which_layers = ()

        for batch_idx in range(p[0].shape[0]):
            b_idx = (targets[:, 0] == batch_idx) # (n_tb,)
            # this_target = ops.masked_select(targets, ops.tile(b_idx[:, None], (1, 6))).view(-1, 6) # (n_tb, 6)
            this_target = ops.masked_select(targets, ops.broadcast_to(b_idx[:, None], b_idx.shape + (6,))).view(-1, 6)  # (n_tb, 6)
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1] # (n_tb, 4)
            txyxy = self.xywh2xyxy(txywh)
            pxyxys = ()
            p_cls = ()
            p_obj = ()
            from_which_layer = ()
            all_b = ()
            all_a = ()
            all_gj = ()
            all_gi = ()
            all_anch = ()

            for i, pi in enumerate(p):
                # b, a, gj, gi = indices[i]
                # b, a, gj, gi = ops.split(indices[i], 0, 4) # ms 1.8.1
                b, a, gj, gi = ops.split(indices[i], 1, 0)  # ms 2.0.0
                b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)

                idx = (b == batch_idx) # (n_tp_b,)
                b, a, gj, gi = ops.masked_select(b, idx), ops.masked_select(a, idx), \
                               ops.masked_select(gj, idx), ops.masked_select(gi, idx)
                # _this_anch = ops.masked_select(anch[i], ops.tile(idx[:, None], (1, 2))).view(-1, 2) # (n_tp_b, 2)
                _this_anch = ops.masked_select(anch[i], ops.broadcast_to(idx[:, None], idx.shape + (2,))).view(-1, 2)  # (n_tp_b, 2)
                all_b += (b,)
                all_a += (a,)
                all_gj += (gj,)
                all_gi += (gi,)
                all_anch += (_this_anch,)
                from_which_layer += (ops.ones((b.shape[0],), ms.int32) * i,)

                fg_pred = pi[b, a, gj, gi] # (n_tp_b, 85)
                p_obj += (fg_pred[:, 4:5],)
                p_cls += (fg_pred[:, 5:],)

                grid = ops.stack((gi, gj), axis=1) # (n_tp_b, 2)
                pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
                pxywh = ops.concat((pxy, pwh), axis=-1) # (n_tp_b, 4)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys += (pxyxy,)

            pxyxys = ops.concat(pxyxys, axis=0) # (n_tp, 4)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = ops.concat(p_obj, axis=0) # (n_tp, 1)
            p_cls = ops.concat(p_cls, axis=0) # (n_tp, 80)
            from_which_layer = ops.concat(from_which_layer, axis=0)
            all_b = ops.concat(all_b, axis=0) # (n_tp,)
            all_a = ops.concat(all_a, axis=0)
            all_gj = ops.concat(all_gj, axis=0)
            all_gi = ops.concat(all_gi, axis=0)
            all_anch = ops.concat(all_anch, axis=0) # (n_tp, 2)

            pair_wise_iou = box_iou(txyxy, pxyxys) # (n_tb, 4), (n_tp, 4) -> (n_tb, n_tp)
            pair_wise_iou_loss = -ops.log(pair_wise_iou + 1e-8)

            v, _ = ops.top_k(pair_wise_iou, min(20, pair_wise_iou.shape[1])) # (n_tb, 10)
            dynamic_ks = v.sum(1).astype(ms.int32).clip(1, None) # (n_tb,)

            gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, 1], ms.int32),
                                           depth=self.nc,
                                           on_value=ops.ones(1, pair_wise_iou.dtype),
                                           off_value=ops.zeros(1, pair_wise_iou.dtype)) # (n_tb, 80)
            # gt_cls_per_image = ops.tile(gt_cls_per_image[:, None, :], (1, pxyxys.shape[0], 1)) # (n_tb, n_tp, 85)
            gt_cls_per_image = ops.broadcast_to(gt_cls_per_image[:, None, :], (gt_cls_per_image.shape[0], pxyxys.shape[0], gt_cls_per_image.shape[1]))  # (n_tb, n_tp, 85)

            num_gt = this_target.shape[0]
            # cls_preds_ = ops.Sigmoid()(ops.tile(p_cls[None, :, :], (num_gt, 1, 1))) * \
            #              ops.Sigmoid()(ops.tile(p_obj[None, :, :], (num_gt, 1, 1))) # (n_tb, n_tp, 80)
            cls_preds_ = ops.Sigmoid()(ops.broadcast_to(p_cls[None, :, :], (num_gt,) + p_cls.shape)) * \
                         ops.Sigmoid()(ops.broadcast_to(p_obj[None, :, :], (num_gt,) + p_obj.shape))  # (n_tb, n_tp, 80)

            y = ops.sqrt(cls_preds_)
            pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
                ops.log(y / (1 - y)),
                gt_cls_per_image,
                ops.ones(1, cls_preds_.dtype),
                ops.ones(1, cls_preds_.dtype),
                reduction="none",
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss  # (n_tb, n_tp)

            # 1. dynamic-k match with pynative and dynamic-shape
            matching_matrix = ops.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = ops.top_k(
                    cost[gt_idx], k=int(dynamic_ks[gt_idx].asnumpy())
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            # 2. dynamic-k match with pynative and static-shape
            # sort_cost, sort_idx = ops.top_k(-cost, 10, sorted=True)
            # sort_cost = -sort_cost
            # pos_idx = ops.stack((mnp.arange(cost.shape[0]), dynamic_ks - 1), -1)
            # pos_v = ops.gather_nd(sort_cost, pos_idx)
            # matching_matrix = ops.cast(cost <= pos_v[:, None], ms.int32)

            anchor_matching_gt = matching_matrix.sum(0)  # (n_tp,)
            anchor_matching_mask = anchor_matching_gt > 1  # (n_tp,)
            anchor_matching_mask_idx = ops.masked_select(mnp.arange(cost.shape[1]), anchor_matching_mask)
            if anchor_matching_mask.astype(cost.dtype).sum() > 0:
                # cost_argmin = ops.argmin(ops.masked_select(cost,
                #                                            ops.tile(anchor_matching_mask[None, :], (cost.shape[0], 1))
                #                                            ).view(cost.shape[0], -1), axis=0)
                cost_argmin = ops.argmin(ops.masked_select(cost,
                                                           ops.broadcast_to(anchor_matching_mask[None, :], (cost.shape[0],) + anchor_matching_mask.shape)
                                                           ).view(cost.shape[0], -1), axis=0)
                # matching_matrix[:, anchor_matching_mask_idx] *= 0.0
                matching_matrix *= 1 - anchor_matching_mask.astype(matching_matrix.dtype)
                matching_matrix[cost_argmin, anchor_matching_mask_idx] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # (n_tp,)

            if fg_mask_inboxes.astype(cost.dtype).sum() > 0.0:
                fg_mask_inboxes_idx = ops.masked_select(mnp.arange(cost.shape[1]), fg_mask_inboxes) # (n_tp_m,)

                # matched_gt_inds = matching_matrix[:, fg_mask_inboxes_idx].argmax(0) # (n_tb, n_tp) -> (n_tb, n_tp_m,) -> (n_tp_m,)
                matched_gt_inds = matching_matrix.transpose(1, 0)[fg_mask_inboxes_idx].argmax(1) # (n_tb, n_tp) -> (n_tp, n_tb) -> (n_tp_m, n_tb) -> (n_tp_m,)

                from_which_layer = from_which_layer[fg_mask_inboxes_idx] # (n_tp,) -> (n_tp_m,)
                all_b = all_b[fg_mask_inboxes_idx]
                all_a = all_a[fg_mask_inboxes_idx]
                all_gj = all_gj[fg_mask_inboxes_idx]
                all_gi = all_gi[fg_mask_inboxes_idx]
                all_anch = all_anch[fg_mask_inboxes_idx]

                this_target = this_target[matched_gt_inds] # (n_tb, 6) -> (n_tp_m, 6)

                matching_from_which_layers += (from_which_layer,)
                matching_bs += (all_b,)
                matching_as += (all_a,)
                matching_gjs += (all_gj,)
                matching_gis += (all_gi,)
                matching_targets += (this_target,)
                matching_anchs += (all_anch,)

        matching_bs = ops.concat(matching_bs, 0)
        matching_as = ops.concat(matching_as, 0)
        matching_gjs = ops.concat(matching_gjs, 0)
        matching_gis = ops.concat(matching_gis, 0)
        matching_targets = ops.concat(matching_targets, 0)
        matching_anchs = ops.concat(matching_anchs, 0)
        matching_from_which_layers = ops.concat(matching_from_which_layers, 0)

        _matching_bs = ()
        _matching_as = ()
        _matching_gjs = ()
        _matching_gis = ()
        _matching_targets = ()
        _matching_anchs = ()
        _matching_from_which_layers = ()

        for i in range(nl):
            layer_mask = matching_from_which_layers == i
            if layer_mask.astype(ms.float16).sum() > 0.0:
                layer_idx = ops.masked_select(mnp.arange(matching_bs.shape[0]), layer_mask)
                _matching_bs += (ops.stop_gradient(matching_bs[layer_idx]),)
                _matching_as += (ops.stop_gradient(matching_as[layer_idx]),)
                _matching_gjs += (ops.stop_gradient(matching_gjs[layer_idx]),)
                _matching_gis += (ops.stop_gradient(matching_gis[layer_idx]),)
                _matching_targets += (ops.stop_gradient(matching_targets[layer_idx]),)
                _matching_anchs += (ops.stop_gradient(matching_anchs[layer_idx]),)
            else:
                _matching_bs += (Tensor([], matching_bs.dtype),)
                _matching_as += (Tensor([], matching_as.dtype),)
                _matching_gjs += (Tensor([], matching_gjs.dtype),)
                _matching_gis += (Tensor([], matching_gis.dtype),)
                _matching_targets += (Tensor([], matching_targets.dtype),)
                _matching_anchs += (Tensor([], matching_anchs.dtype),)

        return _matching_bs, _matching_as, _matching_gjs, _matching_gis, _matching_targets, _matching_anchs

    def build_targets2(self, p, targets, imgs):
        indices, anch = self.find_5_positive(p, targets) # 3 * (4, nm_2), 3 * (nm_2, 2)
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]
        targets = targets.view(-1, 6)  # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0  # (bs*gt_max,)
        targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6)  # (nt, 6)

        matching_bs = ()
        matching_as = ()
        matching_gjs = ()
        matching_gis = ()
        matching_targets = ()
        matching_anchs = ()
        matching_from_which_layers = ()

        for batch_idx in range(p[0].shape[0]):
            b_idx = (targets[:, 0] == batch_idx) # (n_tb,)
            # this_target = ops.masked_select(targets, ops.tile(b_idx[:, None], (1, 6))).view(-1, 6) # (n_tb, 6)
            this_target = ops.masked_select(targets, ops.broadcast_to(b_idx[:, None], b_idx.shape + (6,))).view(-1, 6)  # (n_tb, 6)
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1] # (n_tb, 4)
            txyxy = self.xywh2xyxy(txywh)
            pxyxys = ()
            p_cls = ()
            p_obj = ()
            from_which_layer = ()
            all_b = ()
            all_a = ()
            all_gj = ()
            all_gi = ()
            all_anch = ()

            for i, pi in enumerate(p):
                # b, a, gj, gi = indices[i]
                # b, a, gj, gi = ops.split(indices[i], 0, 4) # ms 1.8.1
                b, a, gj, gi = ops.split(indices[i], 1, 0)  # ms 2.0.0
                b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)

                idx = (b == batch_idx) # (n_tp_b,)
                b, a, gj, gi = ops.masked_select(b, idx), ops.masked_select(a, idx), \
                               ops.masked_select(gj, idx), ops.masked_select(gi, idx)
                # _this_anch = ops.masked_select(anch[i], ops.tile(idx[:, None], (1, 2))).view(-1, 2) # (n_tp_b, 2)
                _this_anch = ops.masked_select(anch[i], ops.broadcast_to(idx[:, None], idx.shape + (2,))).view(-1, 2)  # (n_tp_b, 2)
                all_b += (b,)
                all_a += (a,)
                all_gj += (gj,)
                all_gi += (gi,)
                all_anch += (_this_anch,)
                from_which_layer += (ops.ones((b.shape[0],), ms.int32) * i,)

                fg_pred = pi[b, a, gj, gi] # (n_tp_b, 85)
                p_obj += (fg_pred[:, 4:5],)
                p_cls += (fg_pred[:, 5:],)

                grid = ops.stack((gi, gj), axis=1) # (n_tp_b, 2)
                pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
                pxywh = ops.concat((pxy, pwh), axis=-1) # (n_tp_b, 4)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys += (pxyxy,)

            pxyxys = ops.concat(pxyxys, axis=0) # (n_tp, 4)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = ops.concat(p_obj, axis=0) # (n_tp, 1)
            p_cls = ops.concat(p_cls, axis=0) # (n_tp, 80)
            from_which_layer = ops.concat(from_which_layer, axis=0)
            all_b = ops.concat(all_b, axis=0) # (n_tp,)
            all_a = ops.concat(all_a, axis=0)
            all_gj = ops.concat(all_gj, axis=0)
            all_gi = ops.concat(all_gi, axis=0)
            all_anch = ops.concat(all_anch, axis=0) # (n_tp, 2)

            pair_wise_iou = box_iou(txyxy, pxyxys) # (n_tb, 4), (n_tp, 4) -> (n_tb, n_tp)
            pair_wise_iou_loss = -ops.log(pair_wise_iou + 1e-8)

            v, _ = ops.top_k(pair_wise_iou, min(20, pair_wise_iou.shape[1])) # (n_tb, 10)
            dynamic_ks = v.sum(1).astype(ms.int32).clip(1, None) # (n_tb,)

            gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, 1], ms.int32),
                                           depth=self.nc,
                                           on_value=ops.ones(1, pair_wise_iou.dtype),
                                           off_value=ops.zeros(1, pair_wise_iou.dtype)) # (n_tb, 80)
            # gt_cls_per_image = ops.tile(gt_cls_per_image[:, None, :], (1, pxyxys.shape[0], 1)) # (n_tb, n_tp, 85)
            gt_cls_per_image = ops.broadcast_to(gt_cls_per_image[:, None, :], (gt_cls_per_image.shape[0], pxyxys.shape[0], gt_cls_per_image.shape[1]))  # (n_tb, n_tp, 85)

            num_gt = this_target.shape[0]
            # cls_preds_ = ops.Sigmoid()(ops.tile(p_cls[None, :, :], (num_gt, 1, 1))) * \
            #              ops.Sigmoid()(ops.tile(p_obj[None, :, :], (num_gt, 1, 1))) # (n_tb, n_tp, 80)
            cls_preds_ = ops.Sigmoid()(ops.broadcast_to(p_cls[None, :, :], (num_gt,) + p_cls.shape)) * \
                         ops.Sigmoid()(ops.broadcast_to(p_obj[None, :, :], (num_gt,) + p_obj.shape))  # (n_tb, n_tp, 80)

            y = ops.sqrt(cls_preds_)
            pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
                ops.log(y / (1 - y)),
                gt_cls_per_image,
                ops.ones(1, cls_preds_.dtype),
                ops.ones(1, cls_preds_.dtype),
                reduction="none",
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss  # (n_tb, n_tp)

            # 1. dynamic-k match with pynative and dynamic-shape
            matching_matrix = ops.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = ops.top_k(
                    cost[gt_idx], k=int(dynamic_ks[gt_idx].asnumpy())
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            # 2. dynamic-k match with pynative and static-shape
            # sort_cost, sort_idx = ops.top_k(-cost, 10, sorted=True)
            # sort_cost = -sort_cost
            # pos_idx = ops.stack((mnp.arange(cost.shape[0]), dynamic_ks - 1), -1)
            # pos_v = ops.gather_nd(sort_cost, pos_idx)
            # matching_matrix = ops.cast(cost <= pos_v[:, None], ms.int32)

            anchor_matching_gt = matching_matrix.sum(0)  # (n_tp,)
            anchor_matching_mask = anchor_matching_gt > 1  # (n_tp,)
            anchor_matching_mask_idx = ops.masked_select(mnp.arange(cost.shape[1]), anchor_matching_mask)
            if anchor_matching_mask.astype(cost.dtype).sum() > 0:
                # cost_argmin = ops.argmin(ops.masked_select(cost,
                #                                            ops.tile(anchor_matching_mask[None, :], (cost.shape[0], 1))
                #                                            ).view(cost.shape[0], -1), axis=0)
                cost_argmin = ops.argmin(ops.masked_select(cost,
                                                           ops.broadcast_to(anchor_matching_mask[None, :], (cost.shape[0],) + anchor_matching_mask.shape)
                                                           ).view(cost.shape[0], -1), axis=0)
                # matching_matrix[:, anchor_matching_mask_idx] *= 0.0
                matching_matrix *= 1 - anchor_matching_mask.astype(matching_matrix.dtype)
                matching_matrix[cost_argmin, anchor_matching_mask_idx] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # (n_tp,)

            if fg_mask_inboxes.astype(cost.dtype).sum() > 0.0:
                fg_mask_inboxes_idx = ops.masked_select(mnp.arange(cost.shape[1]), fg_mask_inboxes) # (n_tp_m,)

                # matched_gt_inds = matching_matrix[:, fg_mask_inboxes_idx].argmax(0) # (n_tb, n_tp) -> (n_tb, n_tp_m,) -> (n_tp_m,)
                matched_gt_inds = matching_matrix.transpose(1, 0)[fg_mask_inboxes_idx].argmax(1) # (n_tb, n_tp) -> (n_tp, n_tb) -> (n_tp_m, n_tb) -> (n_tp_m,)

                from_which_layer = from_which_layer[fg_mask_inboxes_idx] # (n_tp,) -> (n_tp_m,)
                all_b = all_b[fg_mask_inboxes_idx]
                all_a = all_a[fg_mask_inboxes_idx]
                all_gj = all_gj[fg_mask_inboxes_idx]
                all_gi = all_gi[fg_mask_inboxes_idx]
                all_anch = all_anch[fg_mask_inboxes_idx]

                this_target = this_target[matched_gt_inds] # (n_tb, 6) -> (n_tp_m, 6)

                matching_from_which_layers += (from_which_layer,)
                matching_bs += (all_b,)
                matching_as += (all_a,)
                matching_gjs += (all_gj,)
                matching_gis += (all_gi,)
                matching_targets += (this_target,)
                matching_anchs += (all_anch,)

        matching_bs = ops.concat(matching_bs, 0)
        matching_as = ops.concat(matching_as, 0)
        matching_gjs = ops.concat(matching_gjs, 0)
        matching_gis = ops.concat(matching_gis, 0)
        matching_targets = ops.concat(matching_targets, 0)
        matching_anchs = ops.concat(matching_anchs, 0)
        matching_from_which_layers = ops.concat(matching_from_which_layers, 0)

        _matching_bs = ()
        _matching_as = ()
        _matching_gjs = ()
        _matching_gis = ()
        _matching_targets = ()
        _matching_anchs = ()
        _matching_from_which_layers = ()

        for i in range(nl):
            layer_mask = matching_from_which_layers == i
            if layer_mask.astype(ms.float16).sum() > 0.0:
                layer_idx = ops.masked_select(mnp.arange(matching_bs.shape[0]), layer_mask)
                _matching_bs += (ops.stop_gradient(matching_bs[layer_idx]),)
                _matching_as += (ops.stop_gradient(matching_as[layer_idx]),)
                _matching_gjs += (ops.stop_gradient(matching_gjs[layer_idx]),)
                _matching_gis += (ops.stop_gradient(matching_gis[layer_idx]),)
                _matching_targets += (ops.stop_gradient(matching_targets[layer_idx]),)
                _matching_anchs += (ops.stop_gradient(matching_anchs[layer_idx]),)
            else:
                _matching_bs += (Tensor([], matching_bs.dtype),)
                _matching_as += (Tensor([], matching_as.dtype),)
                _matching_gjs += (Tensor([], matching_gjs.dtype),)
                _matching_gis += (Tensor([], matching_gis.dtype),)
                _matching_targets += (Tensor([], matching_targets.dtype),)
                _matching_anchs += (Tensor([], matching_anchs.dtype),)

        return _matching_bs, _matching_as, _matching_gjs, _matching_gis, _matching_targets, _matching_anchs

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = ops.zeros(x.shape, x.dtype)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)

        # targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6) # (nt, 6)
        targets = ops.masked_select(targets, ops.broadcast_to(mask_t[:, None], mask_t.shape + (6,))).view(-1, 6)  # (nt, 6)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain

        # ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        ai = ops.broadcast_to(mnp.arange(na, dtype=targets.dtype).view(na, 1), (na, nt))

        # (na, nt, 7)
        # targets = ops.concat((ops.tile(targets[None, :, :], (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # not support dynamic shape
        targets = ops.concat((ops.broadcast_to(targets[None, :, :], (na,) + targets.shape[:]), ai[:, :, None]), 2) # append anchor indices

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            if nt:
                r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio # (na, nt, 2)
                j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7) # (nm, 7)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm, 7)

                # t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy # (nm, 2)
                gxi = gain[[2, 3]] - gxy  # inverse
                jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
                lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
                j, k = jk[:, 0], jk[:, 1] # (nm,)
                l, m = lm[:, 0], lm[:, 1]

                # original
                j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, nm)

                # t = ops.tile(t, (5, 1, 1))  # shape(5, nm, 7)
                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7)  # (nm_2, 7)
                t = ops.broadcast_to(t, (5,) + t.shape)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm_2, 7)

                offsets = ops.zeros_like(gxy)[None, :, :] + off[:, None, :] # (5, nm, 2)
                # offsets = ops.masked_select(offsets, ops.tile(j[:, :, None], (1, 1, 2))).view(-1, 2)
                offsets = ops.masked_select(offsets, ops.broadcast_to(j[:, :, None], j.shape + (2,))).view(-1, 2)
            else:
                t = targets[0]
                offsets = 0
                print("warning: find_3_pos: this batch has no target!")

            # Define # (nm_2,)
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

        return indices, anch

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)

        # targets = ops.masked_select(targets, ops.tile(mask_t[:, None], (1, 6))).view(-1, 6) # (nt, 6)
        targets = ops.masked_select(targets, ops.broadcast_to(mask_t[:, None], mask_t.shape + (6,))).view(-1, 6)  # (nt, 6)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain

        # ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        ai = ops.broadcast_to(mnp.arange(na, dtype=targets.dtype).view(na, 1), (na, nt))

        # (na, nt, 7)
        # targets = ops.concat((ops.tile(targets[None, :, :], (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # not support dynamic shape
        targets = ops.concat((ops.broadcast_to(targets[None, :, :], (na,) + targets.shape[:]), ai[:, :, None]), 2) # append anchor indices

        g = 1.0  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            if nt:
                r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio # (na, nt, 2)
                j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7) # (nm, 7)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm, 7)

                # t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy # (nm, 2)
                gxi = gain[[2, 3]] - gxy  # inverse
                jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
                lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
                j, k = jk[:, 0], jk[:, 1] # (nm,)
                l, m = lm[:, 0], lm[:, 1]

                # original
                j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, nm)

                # t = ops.tile(t, (5, 1, 1))  # shape(5, nm, 7)
                # t = ops.masked_select(t, ops.tile(j[:, :, None], (1, 1, 7))).view(-1, 7)  # (nm_2, 7)
                t = ops.broadcast_to(t, (5,) + t.shape)
                t = ops.masked_select(t, ops.broadcast_to(j[:, :, None], j.shape + (7,))).view(-1, 7)  # (nm_2, 7)

                offsets = ops.zeros_like(gxy)[None, :, :] + off[:, None, :] # (5, nm, 2)
                # offsets = ops.masked_select(offsets, ops.tile(j[:, :, None], (1, 1, 2))).view(-1, 2)
                offsets = ops.masked_select(offsets, ops.broadcast_to(j[:, :, None], j.shape + (2,))).view(-1, 2)
            else:
                t = targets[0]
                offsets = 0
                print("warning: find_3_pos: this batch has no target!")

            # Define # (nm_2,)
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

        return indices, anch

if __name__ == '__main__':
    from pathlib import Path
    from mindspore import context
    from network.yolo import Model
    from utils.general import increment_path
    from utils.config import parse_args

    opt = parse_args("train")
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    opt.total_batch_size = opt.batch_size

    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    model = Model(opt, ch=3, nc=80, anchors=None)
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
