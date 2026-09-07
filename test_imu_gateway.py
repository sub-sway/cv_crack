import io
import json
import time
import unittest
from unittest.mock import patch
from imu_gateway import read_imu_state


class GatewayTests(unittest.TestCase):
    def read(self, data):
        with patch('imu_gateway.urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            return read_imu_state()

    def test_current(self):
        self.assertEqual(self.read({'schema': 1, 'status': 'ready', 'timestamp_unix': time.time()})['status'], 'ready')

    def test_stale(self):
        self.assertEqual(self.read({'schema': 1, 'status': 'ready', 'timestamp_unix': time.time()-10})['status'], 'disconnected')

    def test_invalid_schema(self):
        self.assertEqual(self.read([])['status'], 'disconnected')

    def test_offline(self):
        with patch('imu_gateway.urlopen', side_effect=OSError('offline')):
            self.assertEqual(read_imu_state()['status'], 'disconnected')


if __name__ == '__main__':
    unittest.main()
