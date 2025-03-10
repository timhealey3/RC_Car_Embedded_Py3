import ctypes
import time
Control = ctypes.CDLL("./Control.so")
result = Control.setup()
Control.motorsForward(0);
time.sleep(1);
Control.motorsForward(1);
time.sleep(1);
Control.motorsForward(2);
time.sleep(1);
Control.motorsForward(3);
time.sleep(1)
Control.motorsStop();
Control.cleanup()
print(result)
