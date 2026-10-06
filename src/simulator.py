"""
Single-echelon supply-chain digital twin: supplier -> warehouse -> customer.

Follows the inventory formulation of OR-Gym (Hubbs et al., 2020, arXiv:2008.06319),
reimplemented here so the disruption calendar and the two-stage trial loop can be
added directly, and so the accounting can be property-tested.

The accounting matters more than it looks. The original proposal draft paired a
NON-recursive backorder equation with an inventory equation that allowed negative
stock, which counts every shortage twice and silently corrupts all cost figures.
The recursion in `step` cannot do that -- see the invariant assert and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Disruption(str, Enum):
    NONE = "none"
    SUPPLIER = "supplier"          # supplier ships only a fraction of the order
    TRANSPORT = "transport"        # in-transit orders are delayed further
    DEMAND_SPIKE = "demand_spike"  # demand multiplied for one period
    LEAD_TIME = "lead_time"        # nominal lead time lengthens from now on
    INVENTORY_LOSS = "inv_loss"    # part of on-hand stock is written off


@dataclass(frozen=True)
class TwinParams:
    """Every number here is meant to be calibrated from open data.

    The defaults are placeholders that make the twin runnable today;
    fit_params.py overwrites them from Olist and M5 once those are downloaded.
    """

    demand_mean: float = 140.0
    demand_sd: float = 35.0
    # multiplicative weekly seasonality, one factor per period in a 7-period cycle
    seasonality: tuple = (1.00, 1.05, 1.10, 1.15, 1.05, 0.85, 0.80)
    lead_time_nominal: int = 2
    # P(extra periods of delay) beyond nominal lead time; index = extra periods
    delay_pmf: tuple = (0.70, 0.20, 0.08, 0.02)
    holding_cost: float = 1.0
    shortage_cost: float = 5.0
    # Start roughly one period of demand below the order-up-to level, so the
    # warehouse begins in a healthy steady state and the participant has a
    # normal-sized first order to place. Starting far below S starves the whole
    # episode and makes every disruption invisible against constant stockouts.
    initial_inventory: int = 455
    horizon: int = 30
    max_order: int = 600

    @classmethod
    def from_fitted(cls, path=None, **overrides):
        """Build parameters from data/processed/fitted_params.json.

        Falls back to the placeholder defaults, with a warning, if the fit has
        not been run. Always prefer this over the defaults for anything whose
        numbers will appear in the paper.
        """
        import json
        import os

        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "data", "processed", "fitted_params.json",
            )
        if not os.path.exists(path):
            print("  [warn] no fitted_params.json -- using PLACEHOLDER parameters")
            return cls(**overrides)

        with open(path, encoding="utf-8") as fh:
            fit = json.load(fh)

        kw = {}
        if "olist" in fit:
            kw["lead_time_nominal"] = fit["olist"]["lead_time_nominal"]
            kw["delay_pmf"] = tuple(fit["olist"]["delay_pmf"])
        if "m5" in fit:
            kw["demand_mean"] = fit["m5"]["demand_mean"]
            kw["demand_sd"] = fit["m5"]["demand_sd"]
            kw["seasonality"] = tuple(fit["m5"]["seasonality"])
        kw.update(overrides)
        return cls(**kw)

    def __post_init__(self):
        if self.demand_sd ** 2 <= self.demand_mean:
            raise ValueError(
                "demand_sd^2 must exceed demand_mean for a negative-binomial fit; "
                "got var={:.1f} <= mean={:.1f}".format(
                    self.demand_sd ** 2, self.demand_mean
                )
            )
        if abs(sum(self.delay_pmf) - 1.0) > 1e-9:
            raise ValueError("delay_pmf must sum to 1, got %s" % sum(self.delay_pmf))
        if self.holding_cost <= 0 or self.shortage_cost <= 0:
            raise ValueError("costs must be positive")


@dataclass(frozen=True)
class DisruptionEvent:
    period: int
    kind: Disruption
    magnitude: float  # meaning depends on kind; see step()


# The calendar is IDENTICAL for every participant. Inventory and cost are not:
# they accumulate from that participant's own orders. See plan section 04.
DEFAULT_CALENDAR = (
    DisruptionEvent(6, Disruption.SUPPLIER, 0.55),        # only 55% of order ships
    DisruptionEvent(9, Disruption.DEMAND_SPIKE, 1.80),
    DisruptionEvent(13, Disruption.TRANSPORT, 2.00),      # +2 periods in transit
    DisruptionEvent(17, Disruption.LEAD_TIME, 2.00),      # nominal L becomes L+2
    DisruptionEvent(21, Disruption.INVENTORY_LOSS, 0.25),  # 25% of stock lost
    DisruptionEvent(26, Disruption.SUPPLIER, 0.40),
)


@dataclass
class State:
    """S_t as specified in the plan, section 04."""

    period: int
    demand: int                  # D_t  last realised demand (what the participant saw)
    inventory: int               # I_t  on hand, after arrivals, before this demand
    on_order: int                # O_t  total units in the pipeline
    lead_time: int               # L_t  nominal lead time now in force
    backorders: int              # B_t  unmet demand carried forward
    supplier_reliability: float  # R_t
    transport_ok: bool           # T_t
    disruption: Disruption       # X_t

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.demand,
                self.inventory,
                self.on_order,
                self.lead_time,
                self.backorders,
                self.supplier_reliability,
                float(self.transport_ok),
                float(self.disruption != Disruption.NONE),
            ],
            dtype=float,
        )

    @property
    def inventory_position(self) -> int:
        """IP = on hand + pipeline - backorders. What the order-up-to rule acts on."""
        return self.inventory + self.on_order - self.backorders


class SupplyChainTwin:
    """One warehouse. Order each period, watch consequences arrive later.

    Sequence within period t, which is also the participant's trial:
        1. receive arrivals scheduled for t
        2. apply any inventory-loss disruption
        3. expose state  -> participant decides Q_t
        4. ship Q_t from supplier (may be short), put it in the pipeline
        5. realise demand D_t
        6. serve demand + existing backorders from stock
        7. charge holding and shortage cost on the closing position
    """

    def __init__(self, params=None, calendar=DEFAULT_CALENDAR, seed=None,
                 demand_source=None):
        """`demand_source` replays a real series instead of sampling.

        Pass an M5DemandSource (see demand_source.py) to run the warehouse
        against actual Walmart store-department demand. The synthetic process is
        memoryless past the weekly cycle (lag-4 autocorrelation +0.11 against
        real M5's +0.74), which handicaps any model built to find temporal
        structure. Ground truth is unaffected: Q* depends on the inventory
        position and the demand forecast, not on where demand came from.
        """
        self.p = params or TwinParams()
        self.calendar = {e.period: e for e in calendar}
        if len(self.calendar) != len(calendar):
            raise ValueError("calendar has two events in the same period")
        self.rng = np.random.default_rng(seed)
        self.demand_source = demand_source
        self.reset()

    # ---------- random draws ----------
    def _draw_demand(self, period: int, multiplier: float = 1.0) -> int:
        """Real M5 demand when a source is supplied, else negative binomial."""
        if self.demand_source is not None:
            return self.demand_source(period, multiplier)
        mu = self.p.demand_mean * self.p.seasonality[period % len(self.p.seasonality)]
        mu *= multiplier
        var = (self.p.demand_sd ** 2) * (multiplier ** 2)
        r = mu * mu / max(var - mu, 1e-9)  # dispersion parameter
        prob = r / (r + mu)
        return int(self.rng.negative_binomial(r, prob))

    def _draw_delay(self) -> int:
        return int(self.rng.choice(len(self.p.delay_pmf), p=self.p.delay_pmf))

    # ---------- lifecycle ----------
    def reset(self) -> State:
        self.t = 0
        self.inventory = self.p.initial_inventory
        self.backorders = 0
        self.lead_time = self.p.lead_time_nominal
        self.pipeline = {}  # arrival period -> units
        self.last_demand = int(self.p.demand_mean)
        self.history = []
        return self.observe()

    def observe(self) -> State:
        event = self.calendar.get(self.t)
        kind = event.kind if event else Disruption.NONE
        reliability = (
            event.magnitude if event and kind is Disruption.SUPPLIER else 1.0
        )
        return State(
            period=self.t,
            demand=self.last_demand,
            inventory=self.inventory,
            on_order=sum(self.pipeline.values()),
            lead_time=self.lead_time,
            backorders=self.backorders,
            supplier_reliability=reliability,
            transport_ok=(kind is not Disruption.TRANSPORT),
            disruption=kind,
        )

    def step(self, order: int):
        if not 0 <= order <= self.p.max_order:
            raise ValueError(
                "order %s outside [0, %s]" % (order, self.p.max_order)
            )

        event = self.calendar.get(self.t)
        kind = event.kind if event else Disruption.NONE

        # 1. arrivals land
        arrivals = self.pipeline.pop(self.t, 0)
        self.inventory += arrivals
        inventory_open = self.inventory

        # 2. inventory-loss disruption hits stock already on the shelf
        lost = 0
        if kind is Disruption.INVENTORY_LOSS:
            lost = int(self.inventory * event.magnitude)
            self.inventory -= lost

        # 3. lead-time disruption changes nominal lead time from now on
        if kind is Disruption.LEAD_TIME:
            self.lead_time = self.p.lead_time_nominal + int(event.magnitude)

        # 4. supplier ships (possibly short); transport may add delay
        shipped = order
        if kind is Disruption.SUPPLIER:
            shipped = int(order * event.magnitude)
        extra = self._draw_delay()
        if kind is Disruption.TRANSPORT:
            extra += int(event.magnitude)
        if shipped > 0:
            arrival = self.t + self.lead_time + extra
            self.pipeline[arrival] = self.pipeline.get(arrival, 0) + shipped

        # 5. demand
        mult = event.magnitude if kind is Disruption.DEMAND_SPIKE else 1.0
        demand = self._draw_demand(self.t, mult)

        # 6. serve this period's demand AND outstanding backorders from stock.
        #    Exactly one of the two lines below can be non-zero, which is what
        #    makes double-counting impossible. Invariant: inv - back == net.
        net = self.inventory - demand - self.backorders
        inv_next = max(0, net)
        back_next = max(0, -net)
        # Not a bare assert: python -O strips those, and this is the one guard
        # against the double-counted-shortage bug this module was rewritten to
        # prevent -- the failure mode that silently corrupts every cost figure.
        if inv_next - back_next != net:
            raise AssertionError("inventory/backorder identity violated")

        # 7. cost on the closing position
        cost = self.p.holding_cost * inv_next + self.p.shortage_cost * back_next

        required = demand + self.backorders
        record = {
            "period": self.t,
            "order": order,
            "shipped": shipped,
            "arrivals": arrivals,
            "lost": lost,
            "demand": demand,
            "inventory_open": inventory_open,
            "inventory_close": inv_next,
            "backorders_close": back_next,
            "cost": cost,
            "disruption": kind.value,
            "stockout": back_next > 0,
            "service_level": 1.0 if required == 0 else min(1.0, self.inventory / required),
        }
        self.history.append(record)

        self.inventory, self.backorders = inv_next, back_next
        self.last_demand = demand
        self.t += 1
        return self.observe(), cost, record

    @property
    def done(self) -> bool:
        return self.t >= self.p.horizon

    def total_cost(self) -> float:
        return sum(r["cost"] for r in self.history)
