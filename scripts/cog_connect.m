// Cognionics HD-72 macOS RFCOMM test
// Compile:  clang -framework IOBluetooth -framework Foundation -o cog_connect scripts/cog_connect.m
// Run:      ./cog_connect

#import <Foundation/Foundation.h>
#import <IOBluetooth/IOBluetooth.h>
#include <signal.h>
#include <stdio.h>

#define RFCOMM_CH   1
#define PACKET_SIZE 195
#define SYNC_BYTE   0xFF
#define IMPEDANCE_OFF 0x12

static volatile int g_stop = 0;
static uint64_t     g_packets = 0;
static uint64_t     g_sync_misses = 0;
static NSTimeInterval g_t0 = 0;

static unsigned char g_buf[PACKET_SIZE * 4];
static int           g_buf_len = 0;

static double uv_scale = 1e6 / 4294967296.0;

static void sigint_handler(int s) { g_stop = 1; }

// ── Packet decoder ───────────────────────────────────────────────────────────
static void decode_buf(void) {
    while (g_buf_len >= PACKET_SIZE) {
        // Find sync
        int idx = -1;
        for (int i = 0; i < g_buf_len; i++) {
            if (g_buf[i] == SYNC_BYTE) { idx = i; break; }
        }
        if (idx < 0) { g_sync_misses += g_buf_len; g_buf_len = 0; return; }
        if (idx > 0) { g_sync_misses += idx; memmove(g_buf, g_buf + idx, g_buf_len - idx); g_buf_len -= idx; }
        if (g_buf_len < PACKET_SIZE) return;

        unsigned char *pkt = g_buf;
        uint8_t ctr = pkt[1], st = pkt[2];
        int32_t raw = ((int32_t)pkt[3] << 24) | ((int32_t)pkt[4] << 17) | ((int32_t)pkt[5] << 10);
        double ch0 = raw * uv_scale;

        g_packets++;
        NSTimeInterval elapsed = [NSDate timeIntervalSinceReferenceDate] - g_t0;
        double rate = (elapsed > 0) ? g_packets / elapsed : 0;

        if (g_packets <= 5 || g_packets % 300 == 0) {
            printf("pkt#%6llu  ctr=%3u  st=0x%02X  ch0=%+10.1f uV  rate=%.1f Hz  misses=%llu\n",
                   (unsigned long long)g_packets, ctr, st, ch0, rate,
                   (unsigned long long)g_sync_misses);
            fflush(stdout);
        }

        memmove(g_buf, g_buf + PACKET_SIZE, g_buf_len - PACKET_SIZE);
        g_buf_len -= PACKET_SIZE;
    }
}

// ── RFCOMM delegate ───────────────────────────────────────────────────────────
@interface CogDelegate : NSObject <IOBluetoothRFCOMMChannelDelegate>
@property (strong) IOBluetoothRFCOMMChannel *channel;
@property BOOL opened;
@property BOOL closed;
@end

@implementation CogDelegate

- (void)rfcommChannelOpenComplete:(IOBluetoothRFCOMMChannel *)rfcommChannel
                           status:(IOReturn)error {
    if (error == kIOReturnSuccess) {
        printf("[rfcomm] open complete — streaming\n"); fflush(stdout);
        self.opened = YES;
        uint8_t cmd = IMPEDANCE_OFF;
        [rfcommChannel writeSync:&cmd length:1];
        printf("[rfcomm] sent impedance-off\n"); fflush(stdout);
        g_t0 = [NSDate timeIntervalSinceReferenceDate];
    } else {
        printf("[rfcomm] open failed: 0x%X\n", error); fflush(stdout);
        self.closed = YES;
    }
}

- (void)rfcommChannelData:(IOBluetoothRFCOMMChannel *)rfcommChannel
                     data:(void *)dataPointer
                   length:(size_t)dataLength {
    size_t space = sizeof(g_buf) - g_buf_len;
    size_t copy  = (dataLength < space) ? dataLength : space;
    memcpy(g_buf + g_buf_len, dataPointer, copy);
    g_buf_len += copy;
    decode_buf();
}

- (void)rfcommChannelClosed:(IOBluetoothRFCOMMChannel *)rfcommChannel {
    printf("[rfcomm] channel closed\n"); fflush(stdout);
    self.closed = YES;
    g_stop = 1;
}

@end

// ── main ──────────────────────────────────────────────────────────────────────
int main(void) {
    signal(SIGINT, sigint_handler);

    NSString *addrStr = @"00:12:F3:32:D8:39";
    IOBluetoothDevice *device = [IOBluetoothDevice deviceWithAddressString:addrStr];
    if (!device) {
        fprintf(stderr, "ERROR: device %s not in cache — pair first\n", addrStr.UTF8String);
        return 1;
    }
    printf("Device : %s\n", device.nameOrAddress.UTF8String);
    printf("Status : %s\n", device.isConnected ? "connected" : "not connected");
    printf("Opening RFCOMM channel %d...\n", RFCOMM_CH);
    fflush(stdout);

    CogDelegate *delegate = [CogDelegate new];
    IOBluetoothRFCOMMChannel *channel = nil;

    IOReturn ret = [device openRFCOMMChannelAsync:&channel
                                   withChannelID:RFCOMM_CH
                                        delegate:delegate];
    if (ret != kIOReturnSuccess) {
        printf("openRFCOMMChannelAsync returned 0x%X — headset powered on?\n", ret);
        fflush(stdout);
        return 1;
    }
    printf("Waiting for connection...\n"); fflush(stdout);
    delegate.channel = channel;

    // Run the main run loop until done or timeout
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:15.0];
    while (!g_stop && !delegate.closed) {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
        if (!delegate.opened && [[NSDate date] compare:deadline] == NSOrderedDescending) {
            printf("Timeout waiting for RFCOMM open (15 s).\n");
            fflush(stdout);
            break;
        }
    }

    NSTimeInterval elapsed = [NSDate timeIntervalSinceReferenceDate] - g_t0;
    printf("\n=== summary ===\n");
    printf("packets=%llu  time=%.1fs  rate=%.1f Hz  sync_misses=%llu\n",
           (unsigned long long)g_packets, elapsed,
           elapsed > 0 ? g_packets / elapsed : 0,
           (unsigned long long)g_sync_misses);

    if (channel) [channel closeChannel];
    return 0;
}
