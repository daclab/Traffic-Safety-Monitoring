"""Nonblocking external LEDs through UNO Q's local Arduino Router service.

The MCU sketch owns D7/D8. No App Lab Python package is needed.
"""
import socket
import threading
import time


class RouterClient:
    def __init__(self, address='/var/run/arduino-router.sock'):
        try:
            import msgpack
        except ImportError as e:
            raise RuntimeError('External LEDs require msgpack. Run: bash run.sh --install-led-support') from e
        self.msgpack=msgpack;self.address=address;self.sock=None;self.sequence=0

    def call(self, state):
        if self.sock is None:
            self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.sock.settimeout(.5)
            try:self.sock.connect(self.address)
            except Exception:self.close();raise
            self.unpacker=self.msgpack.Unpacker(raw=False,max_buffer_size=65536)
        self.sequence+=1
        message=[0,self.sequence,'zone_set_leds',[int(state)]]
        try:
            self.sock.sendall(self.msgpack.packb(message,use_bin_type=True))
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline:
                self.sock.settimeout(max(.001,deadline-time.monotonic()))
                chunk=self.sock.recv(4096)
                if not chunk:raise ConnectionError('Arduino Router disconnected')
                self.unpacker.feed(chunk)
                for reply in self.unpacker:
                    if isinstance(reply,list) and len(reply)==4 and reply[0]==1 and reply[1]==self.sequence:
                        if reply[2] is not None:raise RuntimeError('Arduino Router: '+str(reply[2]))
                        if reply[3]!=state:raise RuntimeError('Unexpected LED acknowledgment: '+str(reply[3]))
                        return
            raise TimeoutError('LED acknowledgment timed out')
        except Exception:
            self.close();raise

    def close(self):
        if self.sock is not None:
            self.sock.close();self.sock=None


class ExternalLed:
    def __init__(self,address='/var/run/arduino-router.sock',stale_seconds=2.):
        self.client=RouterClient(address)
        self.condition=threading.Condition();self.stopped=False
        self.state=0;self.updated=time.monotonic();self.stale_seconds=stale_seconds
        self.paths={};self.message='Connecting external LEDs on D7 (red), D8 (green)'
        self.thread=threading.Thread(target=self.worker,name='external-leds',daemon=True)
        self.thread.start()

    def set(self,status):
        state={'DANGER':1,'CLEAR':2}.get(status,0)
        with self.condition:
            changed=state!=self.state
            self.state=state;self.updated=time.monotonic()
            if changed:self.condition.notify()

    def worker(self):
        previous_error=None
        while True:
            with self.condition:
                if self.stopped:break
                state=self.state if time.monotonic()-self.updated<=self.stale_seconds else 0
            try:
                self.client.call(state)
                self.paths={'external':True};self.message='External LEDs: D7 red / D8 green (MCU acknowledged)'
                previous_error=None
            except Exception as e:
                self.paths={};self.message='External LEDs unavailable: '+str(e)+'; upload external_leds.ino and check arduino-router'
                if self.message!=previous_error:print(self.message,flush=True);previous_error=self.message
            with self.condition:
                if not self.stopped:self.condition.wait(.2 if self.paths else 1.)
        try:self.client.call(0)
        except Exception:pass
        self.client.close();self.paths={}

    def close(self):
        with self.condition:self.state=0;self.stopped=True;self.condition.notify()
        self.thread.join(timeout=2.)


def test_leds(address):
    client=RouterClient(address)
    try:
        for state,name in ((1,'RED'),(2,'GREEN'),(0,'BOTH OFF')):
            print('Testing '+name,flush=True)
            for _ in range(10):client.call(state);time.sleep(.2)
        print('MCU acknowledged all three states. Verify the physical colors matched.',flush=True)
    finally:
        try:client.call(0)
        except Exception:pass
        client.close()
