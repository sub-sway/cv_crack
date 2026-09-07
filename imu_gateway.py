"""Read-only local bridge access. No serial dependency in the camera process."""
import json
import math
import time
from urllib.request import urlopen


def read_imu_state():
    try:
        with urlopen('http://127.0.0.1:8766/api/imu', timeout=.4) as response:
            raw = response.read(16385)
        if len(raw) > 16384:
            raise ValueError('Oversized IMU response')
        state = json.loads(raw, parse_constant=lambda _: None)
        if not isinstance(state, dict) or state.get('schema') != 1:
            raise ValueError('Unknown schema')
        if state.get('status') not in ('ready', 'calibrating', 'disconnected'):
            raise ValueError('Invalid status')
        if state['status'] != 'disconnected':
            age = time.time() - float(state.get('timestamp_unix', 0))
            if not math.isfinite(age) or age > 1.5 or age < -2:
                raise ValueError('Stale data')
        return state
    except (OSError, ValueError, TypeError):
        return {'schema': 1, 'status': 'disconnected', 'message': 'IMU 브리지 미연결 — bridge.py 실행 필요'}
