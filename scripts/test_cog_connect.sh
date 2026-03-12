#!/usr/bin/env bash
# Cognionics HD-72 connection test — macOS
# Uses IOBluetooth RFCOMM directly (no TTY needed).
# Usage: bash scripts/test_cog_connect.sh
#
# PREREQUISITES:
#   1. Device is PAIRED in System Settings → Bluetooth
#   2. Device is POWERED ON

set -e
ADDR="00:12:F3:32:D8:39"

echo "==> Cognionics HD-72 connection test"
echo "    addr=$ADDR"
echo "    Press Ctrl-C to stop."
echo ""

python3 -u - "$ADDR" <<'PYEOF'
import sys, os, time, ctypes, threading
import objc
from Foundation import NSObject, NSRunLoop, NSDate

objc.loadBundle('IOBluetooth',
    bundle_path='/System/Library/Frameworks/IOBluetooth.framework',
    module_globals=globals())

ADDR        = sys.argv[1]
RFCOMM_CH   = 1
PACKET_SIZE = 195
SYNC        = 0xFF
UV_SC       = 1e6 / 4_294_967_296.0
IMPEDANCE_OFF = bytes([0x12])

# ─── Shared state ─────────────────────────────────────────────────────────────
open_event   = threading.Event()
open_status  = [None]
buf          = bytearray()
buf_lock     = threading.Lock()
packets      = [0]
sync_misses  = [0]
t0           = [None]
channel_ref  = [None]

# ─── RFCOMM delegate ──────────────────────────────────────────────────────────
class RFDelegate(NSObject):
    def rfcommChannelOpenComplete_status_(self, ch, status):
        open_status[0] = status
        channel_ref[0] = ch
        open_event.set()
        print(f"[rfcomm] open complete  status={status}", flush=True)

    def rfcommChannelData_data_length_(self, ch, data_ptr, length):
        # Copy raw bytes out of the C pointer
        raw = bytes((ctypes.c_uint8 * length).from_address(data_ptr.__index__()))
        with buf_lock:
            buf.extend(raw)

    def rfcommChannelWriteComplete_refcon_status_(self, ch, ref, status):
        if status != 0:
            print(f"[rfcomm] write error status={status}", flush=True)

    def rfcommChannelClosed_(self, ch):
        print(f"[rfcomm] channel closed", flush=True)
        open_event.set()   # unblock if we were waiting

delegate = RFDelegate.alloc().init()

# ─── Open RFCOMM channel ──────────────────────────────────────────────────────
device = IOBluetoothDevice.deviceWithAddressString_(ADDR)
if device is None:
    print(f"ERROR: device {ADDR} not found in cache — pair it first", flush=True)
    sys.exit(1)

print(f"Device : {device.nameOrAddress()}", flush=True)
print(f"Status : {'connected' if device.isConnected() else 'not connected'}", flush=True)
print(f"Opening RFCOMM channel {RFCOMM_CH}...", flush=True)

ch_out = None
ret = device.openRFCOMMChannelAsync_withChannelID_delegate_(
    ch_out,       # channel output (ignored in PyObjC; delegate receives it)
    RFCOMM_CH,
    delegate
)
print(f"openRFCOMMChannelAsync returned {ret}", flush=True)

if ret != 0:
    print(f"ERROR: failed to initiate RFCOMM open (code={ret})", flush=True)
    print("       → Make sure the headset is POWERED ON and try again.", flush=True)
    sys.exit(1)

# Spin the run loop waiting for the open callback (up to 15 s)
deadline = time.time() + 15
print("Waiting for connection (up to 15 s)...", flush=True)
while not open_event.is_set() and time.time() < deadline:
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.3))

if not open_event.is_set() or open_status[0] != 0:
    st = open_status[0]
    print(f"ERROR: RFCOMM did not open (status={st}, timeout={not open_event.is_set()})", flush=True)
    print("       → Is the headset powered on?", flush=True)
    sys.exit(1)

ch = channel_ref[0]
print(f"RFCOMM open!  channel={ch}", flush=True)

# ─── Send impedance-off command ───────────────────────────────────────────────
print(f"Sending impedance-off (0x12)...", flush=True)
data_ns = objc.lookUpClass('NSData').dataWithBytes_length_(IMPEDANCE_OFF, 1)
ch.writeSync_length_(IMPEDANCE_OFF, 1)
print("Sent.", flush=True)
t0[0] = time.time()

# ─── Packet decode loop ───────────────────────────────────────────────────────
print("\nReading packets (Ctrl-C to stop):\n", flush=True)

def decode_loop():
    local_buf = bytearray()
    while True:
        # Grab whatever has arrived
        with buf_lock:
            local_buf.extend(buf)
            buf.clear()

        # Sync + parse
        while True:
            # Find sync byte
            idx = local_buf.find(SYNC)
            if idx < 0:
                sync_misses[0] += len(local_buf)
                local_buf.clear()
                break
            if idx > 0:
                sync_misses[0] += idx
                del local_buf[:idx]
            # Need full packet
            if len(local_buf) < PACKET_SIZE:
                break
            pkt = local_buf[:PACKET_SIZE]
            del local_buf[:PACKET_SIZE]

            counter = pkt[1]
            status  = pkt[2]
            raw = (pkt[3] << 24) | (pkt[4] << 17) | (pkt[5] << 10)
            if raw >= 2**31: raw -= 2**32
            ch0 = raw * UV_SC

            packets[0] += 1
            n = packets[0]
            elapsed = time.time() - t0[0]
            rate = n / elapsed if elapsed > 0 else 0

            if n <= 5 or n % 300 == 0:
                print(f"pkt#{n:6d}  ctr={counter:3d}  st=0x{status:02X}"
                      f"  ch0={ch0:+10.1f} µV  misses={sync_misses[0]}"
                      f"  rate={rate:.1f} Hz", flush=True)

        time.sleep(0.005)   # 5 ms poll

try:
    # Keep run loop spinning so IOBluetooth callbacks keep firing,
    # while the decode thread runs independently.
    decode_thread = threading.Thread(target=decode_loop, daemon=True)
    decode_thread.start()

    while True:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
except KeyboardInterrupt:
    pass

elapsed = time.time() - (t0[0] or time.time())
print(f"\n=== summary ===")
print(f"packets={packets[0]}  time={elapsed:.1f}s"
      f"  rate={packets[0]/max(elapsed,0.001):.1f} Hz"
      f"  sync_misses={sync_misses[0]}")

try:
    channel_ref[0].closeChannel() if channel_ref[0] else None
except Exception:
    pass
PYEOF
