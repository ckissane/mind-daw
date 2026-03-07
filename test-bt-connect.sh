#!/usr/bin/env bash
#
# Repeatedly try to discover, pair, and connect to a Cognionics HD-72
# over Bluetooth Classic (BR/EDR) until successful. Then open an RFCOMM
# serial link and read a few packets to verify data flow.
#
# Usage:  sudo ./test-bt-connect.sh [--max-attempts N]
#
set -euo pipefail

MAX_ATTEMPTS=0  # 0 = infinite
SCAN_SECS=12
RFCOMM_CHANNEL=1
SYNC_BYTE="ff"
PIN="0000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# All log output goes to stderr so $(func) captures only data on stdout
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
fail() { log "ERROR: $*"; }

cleanup() {
    # Release rfcomm device if we bound one
    if [[ -n "${RFCOMM_DEV:-}" ]]; then
        log "Releasing $RFCOMM_DEV"
        rfcomm release "$RFCOMM_DEV" 2>/dev/null || true
    fi
    # Kill any lingering bluetoothctl-agent
    kill "${AGENT_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

# ── Helpers ───────────────────────────────────────────────────────────────────

power_on() {
    bluetoothctl power on >/dev/null 2>&1
    sleep 0.5
}

# Register a BlueZ agent that auto-accepts and supplies PIN 0000.
# bluetoothctl's built-in agent can't auto-supply a PIN, so we use a
# background bluetoothctl process with expect-style feeding.
start_agent() {
    # Set up a default agent that auto-confirms
    bluetoothctl agent off >/dev/null 2>&1 || true
    bluetoothctl agent NoInputNoOutput >/dev/null 2>&1 || true
    bluetoothctl default-agent >/dev/null 2>&1 || true
}

# Scan for a device whose name contains "HD-72" or "Cognionics".
# Prints "AA:BB:CC:DD:EE:FF" on stdout. Log goes to stderr.
scan_for_device() {
    log "Scanning for Cognionics device (${SCAN_SECS}s)..."

    # Start discovery, wait, stop
    timeout "${SCAN_SECS}" bluetoothctl --timeout "$SCAN_SECS" scan on >/dev/null 2>&1 || true
    bluetoothctl scan off >/dev/null 2>&1 || true
    sleep 0.5

    # List all known devices and grep for Cognionics/HD-72
    # Output format: "Device AA:BB:CC:DD:EE:FF SomeName"
    local line
    line=$(bluetoothctl devices 2>/dev/null | grep -iE 'HD-72|Cognionics' | head -1) || true

    if [[ -z "$line" ]]; then
        return 1
    fi

    local addr name
    addr=$(echo "$line" | awk '{print $2}')
    name=$(echo "$line" | cut -d' ' -f3-)
    log "Discovered: $name ($addr)"
    echo "$addr"
}

pair_device() {
    local addr="$1"
    log "Pairing with $addr (PIN: $PIN)..."

    # Try to pair — feed PIN via a heredoc if prompted
    # bluetoothctl pair may prompt "Enter PIN code:" for legacy pairing
    local result
    result=$(echo "$PIN" | bluetoothctl pair "$addr" 2>&1) || true
    log "Pair result: $(echo "$result" | tail -1)"

    sleep 1

    if is_paired "$addr"; then
        log "Pairing successful"
        return 0
    fi

    # Fallback: try with bt-agent if available
    if command -v bt-agent >/dev/null 2>&1; then
        log "Trying bt-agent with PIN $PIN..."
        timeout 10 bt-agent -c NoInputNoOutput -p "$PIN" &
        AGENT_PID=$!
        bluetoothctl pair "$addr" >/dev/null 2>&1 || true
        sleep 2
        kill "$AGENT_PID" 2>/dev/null || true
        AGENT_PID=""
    fi

    is_paired "$addr"
}

trust_device() {
    local addr="$1"
    bluetoothctl trust "$addr" >/dev/null 2>&1
}

connect_device() {
    local addr="$1"
    log "Connecting to $addr..."
    bluetoothctl connect "$addr" >/dev/null 2>&1 || true
    sleep 2
}

is_connected() {
    local addr="$1"
    bluetoothctl info "$addr" 2>/dev/null | grep -q "Connected: yes"
}

is_paired() {
    local addr="$1"
    bluetoothctl info "$addr" 2>/dev/null | grep -q "Paired: yes"
}

# Bind an rfcomm device and try to read packets.
# Returns 0 if we got valid sync bytes.
test_rfcomm() {
    local addr="$1"
    local dev_num=0

    # Find a free rfcomm device number
    while [[ -e "/dev/rfcomm${dev_num}" ]]; do
        dev_num=$((dev_num + 1))
    done
    RFCOMM_DEV="rfcomm${dev_num}"

    log "Binding /dev/$RFCOMM_DEV -> $addr channel $RFCOMM_CHANNEL"
    rfcomm bind "$RFCOMM_DEV" "$addr" "$RFCOMM_CHANNEL" 2>&1 || {
        fail "rfcomm bind failed"
        return 1
    }
    sleep 1

    if [[ ! -e "/dev/$RFCOMM_DEV" ]]; then
        fail "/dev/$RFCOMM_DEV does not exist after bind"
        return 1
    fi

    log "Sending impedance-off command (0x12)..."
    printf '\x12' > "/dev/$RFCOMM_DEV" 2>/dev/null || {
        fail "Failed to write impedance-off command"
        return 1
    }

    log "Reading data from /dev/$RFCOMM_DEV (5s)..."
    local hex_data
    hex_data=$(timeout 5 dd if="/dev/$RFCOMM_DEV" bs=195 count=10 2>/dev/null | xxd -p | tr -d '\n' | head -c 4000) || true

    if [[ -z "$hex_data" ]]; then
        fail "No data received"
        return 1
    fi

    # Count sync bytes (0xFF at packet boundaries — every 195 bytes = 390 hex chars)
    local sync_count=0
    local offset=0
    local packet_hex_len=$((195 * 2))
    while [[ $offset -lt ${#hex_data} ]]; do
        local byte="${hex_data:$offset:2}"
        if [[ "$byte" == "$SYNC_BYTE" ]]; then
            sync_count=$((sync_count + 1))
        fi
        offset=$((offset + packet_hex_len))
    done

    log "Got ${#hex_data} hex chars, $sync_count sync bytes at packet boundaries"

    if [[ $sync_count -ge 2 ]]; then
        log "Valid Cognionics data stream confirmed"
        return 0
    else
        fail "Could not verify packet sync"
        return 1
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

log "=== Cognionics HD-72 Bluetooth Connection Test ==="
if [[ $MAX_ATTEMPTS -gt 0 ]]; then
    log "Max attempts: $MAX_ATTEMPTS"
else
    log "Will retry indefinitely until connection succeeds"
fi
log ""

start_agent

attempt=0
while true; do
    attempt=$((attempt + 1))
    if [[ $MAX_ATTEMPTS -gt 0 && $attempt -gt $MAX_ATTEMPTS ]]; then
        fail "Gave up after $MAX_ATTEMPTS attempts"
        exit 1
    fi

    log "--- Attempt $attempt ---"

    power_on

    # Step 1: Find the device
    ADDR=""
    ADDR=$(scan_for_device) || true
    if [[ -z "$ADDR" ]]; then
        fail "No Cognionics device found, retrying..."
        sleep 3
        continue
    fi

    log "Target: $ADDR"

    # Step 2: Pair if needed
    if ! is_paired "$ADDR"; then
        if ! pair_device "$ADDR"; then
            fail "Pairing failed, retrying..."
            sleep 3
            continue
        fi
    else
        log "Already paired"
    fi

    trust_device "$ADDR"

    # Step 3: Connect
    if ! is_connected "$ADDR"; then
        connect_device "$ADDR"
    fi

    if ! is_connected "$ADDR"; then
        fail "Connection not established, retrying..."
        sleep 3
        continue
    fi

    log "Bluetooth connected"

    # Step 4: Test RFCOMM data stream
    if test_rfcomm "$ADDR"; then
        log ""
        log "=== SUCCESS ==="
        log "Address: $ADDR"
        log "RFCOMM:  /dev/$RFCOMM_DEV"
        log "The EEG is streaming. Run mind-daw to sonify."
        exit 0
    fi

    fail "RFCOMM test failed, retrying..."
    cleanup
    RFCOMM_DEV=""
    sleep 3
done
