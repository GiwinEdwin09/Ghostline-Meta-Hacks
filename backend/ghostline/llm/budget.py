"""The budget guard (build plan Part 23)."""

from __future__ import annotations

from datetime import date

from ghostline.config import Settings

from .errors import DailyBudgetExceeded, ExperimentBudgetExceeded
from .ledger import SpendStore, utc_now


class BudgetGuard:
    """Per-experiment and daily spend caps over a `SpendStore`.

    - `check_call` runs before every live call. Past the experiment cap it raises
      `ExperimentBudgetExceeded`, which the agent loop catches to finish in library-only mode.
    - `check_new_analysis` runs when a job starts. Past the daily cap it raises
      `DailyBudgetExceeded`, whose message is safe to show the user.
    Analyses already running are never cut off by the daily cap, only by their own.
    """

    def __init__(self, store: SpendStore, *, daily_usd: float, per_experiment_usd: float) -> None:
        self.store = store
        self.daily_usd = daily_usd
        self.per_experiment_usd = per_experiment_usd

    @classmethod
    def from_settings(cls, store: SpendStore, settings: Settings) -> BudgetGuard:
        return cls(
            store,
            daily_usd=settings.budget_usd_daily,
            per_experiment_usd=settings.budget_usd_per_experiment,
        )

    def spent_today(self) -> float:
        return self.store.day_total(self._today())

    def experiment_spent(self, experiment_id: str) -> float:
        return self.store.experiment_total(experiment_id)

    def experiment_remaining(self, experiment_id: str) -> float:
        return max(self.per_experiment_usd - self.experiment_spent(experiment_id), 0.0)

    def experiment_exhausted(self, experiment_id: str) -> bool:
        return self.experiment_spent(experiment_id) >= self.per_experiment_usd

    def check_new_analysis(self) -> None:
        spent = self.spent_today()
        if spent >= self.daily_usd:
            raise DailyBudgetExceeded(spent, self.daily_usd)

    def check_call(self, experiment_id: str | None) -> None:
        if experiment_id is None:
            return
        spent = self.experiment_spent(experiment_id)
        if spent >= self.per_experiment_usd:
            raise ExperimentBudgetExceeded(experiment_id, spent, self.per_experiment_usd)

    @staticmethod
    def _today() -> date:
        return utc_now().date()
