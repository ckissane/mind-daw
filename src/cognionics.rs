use std::sync::mpsc;

// ── Constants ────────────────────────────────────────────────────────────────

pub const NUM_CHANNELS: usize = 64;
pub const SAMPLE_RATE: f64 = 300.0;
pub const PACKET_SIZE: usize = 195;
const SYNC_BYTE: u8 = 0xFF;
const IMPEDANCE_OFF_CMD: u8 = 0x12;
const VALID_STATUS: u8 = 0x11;

/// Microvolts conversion factor: 1e6 / 2^32
const UV_SCALE: f64 = 1e6 / 4_294_967_296.0;

// ── Types ────────────────────────────────────────────────────────────────────

/// One decoded Cognionics packet.
pub struct CogSample {
    pub counter: u8,
    pub channels: [f32; NUM_CHANNELS],
    pub status: u8,
}

/// Connection state machine.
#[derive(Clone, Debug)]
pub enum CogState {
    Disconnected,
    Scanning,
    Found { address: [u8; 6], name: String },
    Connecting,
    Streaming,
    Error(String),
}

/// Commands sent from the main thread to the BT worker.
pub enum CogCommand {
    StartScan,
    Connect([u8; 6]),
    Disconnect,
    Shutdown,
}

/// Handle held by the main thread to communicate with the BT worker.
pub struct CogHandle {
    pub cmd_tx: mpsc::Sender<CogCommand>,
    pub sample_rx: mpsc::Receiver<CogSample>,
    pub state_rx: mpsc::Receiver<CogState>,
}

// ── Packet parser ────────────────────────────────────────────────────────────

/// Decode a 195-byte Cognionics wire-format packet.
///
/// Layout: [sync(1)] [counter(1)] [status(1)] [64 channels × 3 bytes(192)]
/// Each channel is 3 bytes, 7-bit encoded:
///   raw_i32 = (MSB << 24) | (LSB2 << 17) | (LSB1 << 10)
///   microvolts = raw_i32 as f64 * UV_SCALE
pub fn parse_packet(buf: &[u8; PACKET_SIZE]) -> Option<CogSample> {
    if buf[0] != SYNC_BYTE {
        return None;
    }

    let counter = buf[1];
    let status = buf[2];

    let mut channels = [0.0f32; NUM_CHANNELS];
    for ch in 0..NUM_CHANNELS {
        let offset = 3 + ch * 3;
        let msb = buf[offset] as i32;
        let lsb2 = buf[offset + 1] as i32;
        let lsb1 = buf[offset + 2] as i32;

        let raw = (msb << 24) | (lsb2 << 17) | (lsb1 << 10);
        channels[ch] = (raw as f64 * UV_SCALE) as f32;
    }

    Some(CogSample {
        counter,
        channels,
        status,
    })
}

// ── BT worker ────────────────────────────────────────────────────────────────

/// Spawn the Cognionics Bluetooth worker on a dedicated OS thread.
///
/// Returns a `CogHandle` for sending commands and receiving samples/state
/// from the main thread. The worker runs its own Tokio current-thread runtime
/// to drive bluer's async Bluetooth API.
pub fn spawn_cog_worker() -> CogHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<CogCommand>();
    let (sample_tx, sample_rx) = mpsc::sync_channel::<CogSample>(1024);
    let (state_tx, state_rx) = mpsc::channel::<CogState>();

    std::thread::Builder::new()
        .name("cog-bt-worker".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime for BT worker");

            rt.block_on(worker_loop(cmd_rx, sample_tx, state_tx));
        })
        .expect("failed to spawn BT worker thread");

    CogHandle {
        cmd_tx,
        sample_rx,
        state_rx,
    }
}

/// Send a state update, ignoring disconnected receiver.
fn send_state(tx: &mpsc::Sender<CogState>, state: CogState) {
    let _ = tx.send(state);
}

async fn worker_loop(
    cmd_rx: mpsc::Receiver<CogCommand>,
    sample_tx: mpsc::SyncSender<CogSample>,
    state_tx: mpsc::Sender<CogState>,
) {
    loop {
        // Block waiting for the next command
        let cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break, // main thread dropped the sender
        };

        match cmd {
            CogCommand::StartScan => {
                send_state(&state_tx, CogState::Scanning);
                match scan_for_device().await {
                    Ok((addr, name)) => {
                        send_state(
                            &state_tx,
                            CogState::Found {
                                address: addr,
                                name,
                            },
                        );
                    }
                    Err(e) => {
                        send_state(&state_tx, CogState::Error(format!("Scan failed: {e}")));
                    }
                }
            }
            CogCommand::Connect(addr) => {
                send_state(&state_tx, CogState::Connecting);
                match connect_and_stream(addr, &cmd_rx, &sample_tx, &state_tx).await {
                    Ok(()) => {
                        send_state(&state_tx, CogState::Disconnected);
                    }
                    Err(e) => {
                        send_state(
                            &state_tx,
                            CogState::Error(format!("Connection error: {e}")),
                        );
                    }
                }
            }
            CogCommand::Disconnect => {
                // Already disconnected if we're here
                send_state(&state_tx, CogState::Disconnected);
            }
            CogCommand::Shutdown => break,
        }
    }
}

/// Scan for a Cognionics HD-72 device via Bluetooth Classic (BR/EDR).
async fn scan_for_device() -> Result<([u8; 6], String), bluer::Error> {
    use bluer::AdapterEvent;
    use futures::StreamExt;

    let session = bluer::Session::new().await?;
    let adapter = session.default_adapter().await?;
    adapter.set_powered(true).await?;

    // Set discovery filter to BR/EDR only
    adapter
        .set_discovery_filter(bluer::DiscoveryFilter {
            transport: bluer::DiscoveryTransport::BrEdr,
            ..Default::default()
        })
        .await?;

    let events = adapter.discover_devices().await?;
    tokio::pin!(events);

    let timeout = tokio::time::sleep(std::time::Duration::from_secs(15));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            Some(event) = events.next() => {
                if let AdapterEvent::DeviceAdded(addr) = event {
                    let device = adapter.device(addr)?;
                    if let Ok(Some(name)) = device.name().await {
                        if name.contains("HD-72") || name.contains("Cognionics") {
                            let addr_bytes = addr.0;
                            return Ok((addr_bytes, name));
                        }
                    }
                }
            }
            () = &mut timeout => {
                return Err(bluer::Error {
                    kind: bluer::ErrorKind::Failed,
                    message: "Scan timed out — no Cognionics device found".into(),
                });
            }
        }
    }
}

/// Connect to the device, send impedance-off command, and enter the read loop.
/// Returns `Ok(())` when cleanly disconnected via `CogCommand::Disconnect`.
async fn connect_and_stream(
    addr: [u8; 6],
    cmd_rx: &mpsc::Receiver<CogCommand>,
    sample_tx: &mpsc::SyncSender<CogSample>,
    state_tx: &mpsc::Sender<CogState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let session = bluer::Session::new().await?;
    let adapter = session.default_adapter().await?;
    let bt_addr = bluer::Address(addr);
    let device = adapter.device(bt_addr)?;

    // Ensure device is paired
    if !device.is_paired().await? {
        device.pair().await?;
    }

    // Connect via RFCOMM (channel 1 is typical for SPP)
    let mut stream = bluer::rfcomm::Stream::connect(bluer::rfcomm::SocketAddr {
        addr: bt_addr,
        channel: 1,
    })
    .await?;

    // Send impedance-off command
    stream.write_all(&[IMPEDANCE_OFF_CMD]).await?;
    stream.flush().await?;

    send_state(state_tx, CogState::Streaming);

    // Read loop: sync-byte alignment then full packets
    let mut packet_buf = [0u8; PACKET_SIZE];
    let mut single = [0u8; 1];

    loop {
        // Check for disconnect command (non-blocking)
        match cmd_rx.try_recv() {
            Ok(CogCommand::Disconnect) | Ok(CogCommand::Shutdown) => break,
            _ => {}
        }

        // Sync: read byte-by-byte until we find the sync byte
        loop {
            stream.read_exact(&mut single).await?;
            if single[0] == SYNC_BYTE {
                packet_buf[0] = SYNC_BYTE;
                break;
            }
        }

        // Read the remaining 194 bytes
        stream.read_exact(&mut packet_buf[1..]).await?;

        // Parse and send
        if let Some(sample) = parse_packet(&packet_buf) {
            // If the channel is full, drop the sample (backpressure)
            let _ = sample_tx.try_send(sample);
        }
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_packet() {
        let mut buf = [0u8; PACKET_SIZE];
        buf[0] = SYNC_BYTE;
        buf[1] = 42; // counter
        buf[2] = VALID_STATUS; // status

        // Set channel 0 to a known value: MSB=0x40, LSB2=0x00, LSB1=0x00
        // raw = (0x40 << 24) | (0x00 << 17) | (0x00 << 10) = 0x40000000 = 1073741824
        // uv = 1073741824 * (1e6 / 2^32) ≈ 250000.0
        buf[3] = 0x40;
        buf[4] = 0x00;
        buf[5] = 0x00;

        let sample = parse_packet(&buf).unwrap();
        assert_eq!(sample.counter, 42);
        assert_eq!(sample.status, VALID_STATUS);
        assert!((sample.channels[0] - 250000.0).abs() < 1.0);
        // All other channels should be 0
        assert_eq!(sample.channels[1], 0.0);
    }

    #[test]
    fn parse_bad_sync_returns_none() {
        let mut buf = [0u8; PACKET_SIZE];
        buf[0] = 0x00; // wrong sync byte
        assert!(parse_packet(&buf).is_none());
    }
}
