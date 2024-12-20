import unittest

from src.car import RC_Car


class MyTestCase(unittest.TestCase):
    def test_car_controls(self):
        car = RC_Car()

if __name__ == '__main__':
    unittest.main()
