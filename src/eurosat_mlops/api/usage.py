"""Per-user usage tracking in SQL (SQLite locally, Postgres in compose and production)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import DateTime, Float, Integer, String, case, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from eurosat_mlops.drift.stats import FEATURES


class Base(DeclarativeBase):
    pass


class UsageRow(Base):
    __tablename__ = "usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    user: Mapped[str] = mapped_column(String(64), index=True)
    endpoint: Mapped[str] = mapped_column(String(32))
    status: Mapped[int] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    contrast: Mapped[float | None] = mapped_column(Float, nullable=True)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite") and ":memory:" in url:
        return create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


class UsageStore:
    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)
        Base.metadata.create_all(self.engine)

    def record(
        self,
        user: str,
        endpoint: str,
        status: int,
        latency_ms: float,
        label: str | None = None,
        confidence: float | None = None,
        stats: dict[str, float] | None = None,
    ) -> None:
        row = UsageRow(
            ts=datetime.now(UTC),
            user=user,
            endpoint=endpoint,
            status=status,
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            **(stats or {}),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def summary(self, user: str | None = None) -> list[dict]:
        """Requests, errors, mean latency and last call, grouped by user."""
        q = select(
            UsageRow.user,
            func.count(UsageRow.id),
            func.sum(case((UsageRow.status >= 400, 1), else_=0)),
            func.avg(UsageRow.latency_ms),
            func.max(UsageRow.ts),
        ).group_by(UsageRow.user)
        if user:
            q = q.where(UsageRow.user == user)
        with Session(self.engine) as session:
            rows = session.execute(q.order_by(UsageRow.user)).all()
        return [
            {
                "user": r[0],
                "requests": int(r[1]),
                "errors": int(r[2] or 0),
                "mean_latency_ms": round(float(r[3] or 0), 2),
                "last_call": r[4].isoformat() if r[4] else None,
            }
            for r in rows
        ]

    def recent_stats(self, n: int) -> dict[str, np.ndarray]:
        """Image statistics of the last n successful predictions, for drift checks."""
        q = (
            select(*(getattr(UsageRow, f) for f in FEATURES))
            .where(UsageRow.endpoint == "predict", UsageRow.brightness.is_not(None))
            .order_by(UsageRow.id.desc())
            .limit(n)
        )
        with Session(self.engine) as session:
            rows = session.execute(q).all()
        cols = list(zip(*rows, strict=True)) if rows else [[] for _ in FEATURES]
        return {f: np.asarray(c, dtype=np.float32) for f, c in zip(FEATURES, cols, strict=True)}
