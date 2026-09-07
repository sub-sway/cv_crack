"""Collect mounted raw magnetometer data from a RUNNING bridge (no second serial reader)."""
import argparse
import json
import math
import time
from urllib.request import urlopen

from imu_core import vector


def fit(samples, axes):
    if len(samples) < 200:
        raise ValueError('Not enough magnetic samples (need at least 200)')
    lo = [min(s[i] for s in samples) for i in range(3)]
    hi = [max(s[i] for s in samples) for i in range(3)]
    radius = [(hi[i]-lo[i])/2 for i in range(3)]
    if min(radius) < 10 or max(radius)/min(radius) > 4:
        raise ValueError('Insufficient 3D rotation or severe field distortion; repeat calibration')
    field = sum(radius)/3
    if not 15 < field < 80:
        raise ValueError('Implausible magnetic field; move away from metal/motors')
    return {'schema': 1, 'axes': axes, 'offset_uT': [(hi[i]+lo[i])/2 for i in range(3)],
            'scale': [field/r for r in radius], 'field_uT': field, 'samples': len(samples),
            'method': 'minmax_hard_iron_diagonal_soft_iron_prototype'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--output', default='mag_calibration.json')
    p.add_argument('--http-port', type=int, default=8766)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 10 <= args.seconds <= 600:
        p.error('--seconds must be between 10 and 600')
    print('Rotate slowly about ALL THREE axes, away from steel/magnets. Never move a powered robot unsafely.', flush=True)
    samples, axes, previous = [], None, None
    deadline = time.monotonic()+args.seconds
    while time.monotonic() < deadline:
        with urlopen(f'http://127.0.0.1:{args.http_port}/api/imu', timeout=1) as response:
            data = json.load(response)
        if data.get('demo'):
            raise ValueError('Cannot calibrate synthetic demo data')
        if data.get('status') == 'disconnected':
            raise ValueError('Bridge disconnected')
        current_axes = data.get('axes')
        if axes is not None and current_axes != axes:
            raise ValueError('Mounting axes changed during calibration')
        axes = current_axes
        sample_id = (data.get('epoch'), data.get('seq'))
        if sample_id != previous and data.get('mag_uT') is not None:
            samples.append(vector(data['mag_uT']))
        previous = sample_id
        time.sleep(.04)
    calibration = fit(samples, axes)
    # Do not silently overwrite a previous calibration.
    with open(args.output, 'x', encoding='utf-8') as output:
        json.dump(calibration, output, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Saved {len(samples)} samples to {args.output}. Restart bridge with --mag-cal {args.output}')


if __name__ == '__main__':
    main()
