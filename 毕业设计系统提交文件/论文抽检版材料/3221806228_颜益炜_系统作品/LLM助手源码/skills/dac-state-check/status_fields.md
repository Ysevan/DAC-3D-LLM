# Runtime Status Fields

- `state`: idle, queued, running, completed, stopped, or error.
- `progress`: integer percentage when available.
- `message`: operator-facing runtime message.
- `step`: current processing step.
- `updated_at`: timestamp from DAC host.
- `latest_result`: latest parsed result when host publishes it.
- `result_history`: recent sample result history.
