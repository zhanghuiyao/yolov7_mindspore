from mindspore import nn, ops

from .conv import Conv
from .pool import PoolWithPad

class SPPCSPC(nn.Cell):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.CellList([PoolWithPad(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def construct(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        m_tuple = (x1,)
        for i in range(len(self.m)):
            m_tuple += (self.m[i](x1),)
        y1 = self.cv6(self.cv5(ops.Concat(axis=1)(m_tuple)))
        y2 = self.cv2(x)
        return self.cv7(ops.Concat(axis=1)((y1, y2)))
