"""Teensy serial -> processor -> localhost HTTP, shared by dashboard and ROS2."""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from imu_core import G, Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='Teensy USB port, e.g. COM5')
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Synthetic rotation only, not a sensor measurement')
    parser.add_argument('--http-port', type=int, default=8766)
    parser.add_argument('--axes', default='1,2,3', help='Body X,Y,Z from signed sensor axes, e.g. 2,-1,3')
    parser.add_argument('--log', help='Append processed samples to this JSONL file')
    parser.add_argument('--mag-cal', help='Opt in to magnetic heading with a calibration JSON file')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Skip the stationary hold. Heading drifts and tilt sits a few degrees off; '
                             'readings are flagged and are not measurements.')
    args = parser.parse_args()
    if args.list_ports:
        from serial.tools import list_ports
        for item in list_ports.comports():
            print(item.device, item.description)
        return
    if bool(args.demo) == bool(args.port):
        parser.error('Choose exactly one of --demo or --port COMx')
    serial = None
    if not args.demo:
        try:
            import serial
        except ImportError:
            parser.error('Install pyserial first: python -m pip install -r requirements.txt')
    calibration = None
    if args.mag_cal:
        with open(args.mag_cal, encoding='utf-8') as source:
            calibration = json.load(source)
    processor = Processor(tuple(int(a) for a in args.axes.split(',')), mag_calibration=calibration,
                          skip_calibration=args.no_calibration)
    lock, stop = threading.Lock(), threading.Event()
    log = open(args.log, 'a', encoding='utf-8', buffering=1) if args.log else None

    def accept(packet):
        with lock:
            state = dict(processor.update(packet))
        state['demo'] = args.demo
        if log:
            log.write(json.dumps(state, ensure_ascii=False, allow_nan=False)+'\n')

    def reader():
        seq = 0
        while not stop.is_set():
            if args.demo:
                angle = max(0, seq-249)*.01*math.pi/12
                accept({'v': 1, 'type': 'imu', 'seq': seq & 0xffffffff,
                        't_us': (seq*10000) & 0xffffffff, 'chip': 'MPU9250', 'mag_present': True,
                        'accel_mps2': [0., 0., G], 'gyro_rps': [0., 0., 0. if seq < 250 else -math.pi/12],
                        'mag_uT': [40*math.cos(angle), 40*math.sin(angle), -20.]})
                seq += 1
                stop.wait(.01)
                continue
            try:
                with serial.Serial(args.port, 115200, timeout=.2) as device:
                    device.reset_input_buffer()
                    with lock:
                        processor.reset()
                    print('Connected:', args.port, 'Keep robot STOPPED for calibration.', flush=True)
                    pending = b''
                    discard = False
                    while not stop.is_set():
                        chunk = device.read_until(b'\n', 2048)
                        if not chunk:
                            continue
                        if discard:
                            if chunk.endswith(b'\n'):
                                discard = False
                            continue
                        pending += chunk
                        if len(pending) > 4096:
                            pending = b''
                            discard = not chunk.endswith(b'\n')
                            continue
                        if not pending.endswith(b'\n'):
                            continue
                        line, pending = pending, b''
                        try:
                            packet = json.loads(line)
                            if isinstance(packet, dict) and packet.get('type') == 'status':
                                print(packet.get('message', 'Device status'), flush=True)
                                with lock:
                                    processor.reset()
                            else:
                                accept(packet)
                        except (ValueError, TypeError, UnicodeError):
                            continue
            except (serial.SerialException, OSError) as exc:
                with lock:
                    processor.reset()
                print('Serial unavailable; retry in 2s:', exc, flush=True)
                stop.wait(2)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/api/imu':
                self.send_error(404)
                return
            with lock:
                state = processor.snapshot()
            state['demo'] = args.demo
            body = json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.http_port), Handler)
    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    print(f'IMU API: http://127.0.0.1:{args.http_port}/api/imu', flush=True)
    print('DEMO: synthetic data' if args.demo else 'Relative heading; experimental inertial speed/position', flush=True)
    if args.no_calibration:
        print('보정 생략 모드: 자이로 바이어스 미보정 — 방위/기울기는 참고용, 측정값 아님', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=3)
        if log and not worker.is_alive():
            log.close()


if __name__ == '__main__':
    main()
