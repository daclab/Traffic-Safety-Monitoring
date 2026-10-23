import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from external_leds import ExternalLed,RouterClient


class ExternalLedTests(unittest.TestCase):
    def test_router_packets_fragmented_response_and_error(self):
        seen=[]
        class Wire:
            def settimeout(self,value):pass
            def connect(self,path):self.path=path
            def sendall(self,data):
                req=msgpack.unpackb(data,raw=False);seen.append(req)
                error='test MCU missing' if len(seen)==3 else None
                reply=msgpack.packb([1,req[1],error,req[3][0]])
                self.chunks=[reply[:2],reply[2:]]
            def recv(self,size):return self.chunks.pop(0)
            def close(self):pass
        with patch('external_leds.socket.socket',return_value=Wire()):
            c=RouterClient('/var/run/arduino-router.sock')
            try:
                c.call(1);c.call(2)
                with self.assertRaisesRegex(RuntimeError,'test MCU missing'):c.call(0)
                self.assertEqual([r[3][0] for r in seen],[1,2,0])
                self.assertTrue(all(r[0]==0 and r[2]=='zone_set_leds' for r in seen))
                self.assertIsNone(c.sock)
            finally:c.close()

    def test_states_heartbeat_stale_and_close(self):
        seen=[]
        class Client:
            def __init__(self,*a):pass
            def call(self,state):seen.append(state)
            def close(self):pass
        def wait_for(state):
            deadline=time.monotonic()+2
            while (not seen or seen[-1]!=state) and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(seen[-1],state)
        with patch('external_leds.RouterClient',Client):
            led=ExternalLed(stale_seconds=.25)
            try:
                wait_for(0);led.set('DANGER');wait_for(1)
                led.set('CLEAR');wait_for(2)
                # A worker heartbeat cannot keep green alive without fresh decisions.
                time.sleep(.55);self.assertEqual(seen[-1],0)
                led.set('DANGER');wait_for(1);led.set('CALIBRATE');wait_for(0)
                led.set('DANGER');wait_for(1)
            finally:led.close()
            self.assertFalse(led.thread.is_alive());self.assertEqual(seen[-1],0)


if __name__=='__main__':unittest.main()
