import unittest
import sys
sys.path.append('/home/timh/codingProjects/src/car')
import RC_Car

class MyTestCase(unittest.TestCase):
    def test_car_controls(self):
        #TODO write tests
        car = RC_Car()
        
if __name__ == '__main__':
    unittest.main()
