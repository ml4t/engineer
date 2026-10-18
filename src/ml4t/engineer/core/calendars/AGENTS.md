# calendars/ - Trading Calendar Helpers

Calendar utilities for session-aware workflows.

## Files

| File | Purpose |
|------|---------|
| `base.py` | common calendar interfaces |
| `equity.py` | equity-market calendar behavior |
| `crypto.py` | always-open crypto calendar behavior |

## Notes

- `EquityCalendar(calendar=...)` requires the authoritative schedule provider. Never
  substitute weekdays, fixed US hours or a different exchange after a provider error.
- Naive timestamps use the selected exchange timezone unless explicitly overridden.
  Keep membership, boundary lookup, enumeration and duration consistent.
- Feature grids and session-aware labeling reuse `labeling.calendar.PandasMarketCalendar`.
  This does not imply that all labeling APIs already consume the shared temporal contract.
- Calendar-free feature calculations use `compute_features(..., calendar=None)`.
  A named continuous calendar is not calendar-free execution.
- `CryptoCalendar` uses UTC daily sessions, not hourly observation points. Maintenance
  reopens do not create extra sessions; keep membership and daily boundary methods distinct.
