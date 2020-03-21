from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

class LgFilter(bt.Indicator):
    alias = ('L0', 'L1', 'L2', 'L3', 'Filt', 'RSI')
    lines = ('l0', 'l1', 'l2', 'l3', 'filt', 'rsi')
    plotinfo  = dict(subplot=False)
    plotlines = dict(
        l0=dict(_plotskip='True'),
        l1=dict(_plotskip='True'),
        l2=dict(_plotskip='True'),
        l3=dict(_plotskip='True'),
        rsi=dict(_plotskip='True'),
    )

    params = (('gamma', 0.3),)

    def next(self):
        prev_l0 = (self.l.l0[-1] if len(self.l.l0) > 1 else 0.0)
        prev_l1 = (self.l.l1[-1] if len(self.l.l1) > 1 else 0.0)
        prev_l2 = (self.l.l2[-1] if len(self.l.l2) > 1 else 0.0)
        prev_l3 = (self.l.l3[-1] if len(self.l.l3) > 1 else 0.0)

        self.l.l0[0] = (1 - self.p.gamma) * self.data[0] + self.p.gamma * prev_l0
        self.l.l1[0] = -self.p.gamma * self.l.l0[0] + prev_l0 + self.p.gamma * prev_l1
        self.l.l2[0] = -self.p.gamma * self.l.l1[0] + prev_l1 + self.p.gamma * prev_l2
        self.l.l3[0] = -self.p.gamma * self.l.l2[0] + prev_l2 + self.p.gamma * prev_l3

        self.l.filt[0] = (self.l.l0[0] + 2 * self.l.l1[0] + 2 * self.l.l2[0] + self.l.l3[0]) / 6
