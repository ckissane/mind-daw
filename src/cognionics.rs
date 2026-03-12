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
    Found { id: String, name: String },
    Connecting,
    Streaming,
    Error(String),
}

/// Commands sent from the main thread to the BT worker.
pub enum CogCommand {
    StartScan,
    Connect(String),
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
#[cfg(target_os = "linux")]
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

/// Stub implementation for unsupported platforms.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn spawn_cog_worker() -> CogHandle {
    let (cmd_tx, _cmd_rx) = mpsc::channel::<CogCommand>();
    let (_sample_tx, sample_rx) = mpsc::sync_channel::<CogSample>(1);
    let (_state_tx, state_rx) = mpsc::channel::<CogState>();
    CogHandle {
        cmd_tx,
        sample_rx,
        state_rx,
    }
}

/// Send a state update, ignoring disconnected receiver.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn send_state(tx: &mpsc::Sender<CogState>, state: CogState) {
    eprintln!("[cog] state -> {state:?}");
    let _ = tx.send(state);
}

#[cfg(target_os = "linux")]
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
                    Ok((addr_str, name)) => {
                        send_state(
                            &state_tx,
                            CogState::Found {
                                id: addr_str,
                                name,
                            },
                        );
                    }
                    Err(e) => {
                        send_state(&state_tx, CogState::Error(format!("Scan failed: {e}")));
                    }
                }
            }
            CogCommand::Connect(addr_str) => {
                send_state(&state_tx, CogState::Connecting);
                let addr: bluer::Address = match addr_str.parse() {
                    Ok(a) => a,
                    Err(e) => {
                        send_state(
                            &state_tx,
                            CogState::Error(format!("Bad address: {e}")),
                        );
                        continue;
                    }
                };
                match connect_and_stream(addr.0, &cmd_rx, &sample_tx, &state_tx).await {
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
#[cfg(target_os = "linux")]
async fn scan_for_device() -> Result<(String, String), bluer::Error> {
    use bluer::AdapterEvent;
    use futures::StreamExt;

    eprintln!("[cog] scan: creating BlueZ session...");
    let session = bluer::Session::new().await?;
    let adapter = session.default_adapter().await?;
    eprintln!("[cog] scan: adapter = {}", adapter.name());
    adapter.set_powered(true).await?;

    // Set discovery filter to BR/EDR only
    eprintln!("[cog] scan: setting BR/EDR discovery filter...");
    adapter
        .set_discovery_filter(bluer::DiscoveryFilter {
            transport: bluer::DiscoveryTransport::BrEdr,
            ..Default::default()
        })
        .await?;

    eprintln!("[cog] scan: starting discovery (15s timeout)...");
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
                        eprintln!("[cog] scan: found device {addr} name={name:?}");
                        if name.contains("HD-72") || name.contains("Cognionics") {
                            eprintln!("[cog] scan: matched Cognionics device!");
                            return Ok((addr.to_string(), name));
                        }
                    } else {
                        eprintln!("[cog] scan: found device {addr} (no name)");
                    }
                }
            }
            () = &mut timeout => {
                eprintln!("[cog] scan: timed out");
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
#[cfg(target_os = "linux")]
async fn connect_and_stream(
    addr: [u8; 6],
    cmd_rx: &mpsc::Receiver<CogCommand>,
    sample_tx: &mpsc::SyncSender<CogSample>,
    state_tx: &mpsc::Sender<CogState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let addr_fmt = format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        addr[0], addr[1], addr[2], addr[3], addr[4], addr[5]
    );
    eprintln!("[cog] connect: target {addr_fmt}");

    eprintln!("[cog] connect: creating BlueZ session...");
    let session = bluer::Session::new().await?;
    let adapter = session.default_adapter().await?;
    let bt_addr = bluer::Address(addr);
    let device = adapter.device(bt_addr)?;

    // Ensure device is paired
    let paired = device.is_paired().await?;
    eprintln!("[cog] connect: paired={paired}");
    if !paired {
        eprintln!("[cog] connect: initiating pairing...");
        device.pair().await?;
        eprintln!("[cog] connect: pairing complete");
    }

    // Disconnect any existing OS-level connection (avoids EBUSY on RFCOMM)
    let connected = device.is_connected().await.unwrap_or(false);
    if connected {
        eprintln!("[cog] connect: device already connected at OS level, disconnecting first...");
        let _ = device.disconnect().await;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        eprintln!("[cog] connect: OS-level disconnect done");
    }

    // Connect via RFCOMM (channel 1 is typical for SPP)
    eprintln!("[cog] connect: opening RFCOMM channel {RFCOMM_CHANNEL}...", RFCOMM_CHANNEL = 1);
    let mut stream = bluer::rfcomm::Stream::connect(bluer::rfcomm::SocketAddr {
        addr: bt_addr,
        channel: 1,
    })
    .await?;
    eprintln!("[cog] connect: RFCOMM connected");

    // Send impedance-off command
    eprintln!("[cog] connect: sending impedance-off (0x{IMPEDANCE_OFF_CMD:02X})...");
    stream.write_all(&[IMPEDANCE_OFF_CMD]).await?;
    stream.flush().await?;
    eprintln!("[cog] connect: impedance-off sent, entering read loop");

    send_state(state_tx, CogState::Streaming);

    // Read loop: sync-byte alignment then full packets
    let mut packet_buf = [0u8; PACKET_SIZE];
    let mut single = [0u8; 1];
    let mut packet_count: u64 = 0;
    let mut sync_miss: u64 = 0;

    loop {
        // Check for disconnect command (non-blocking)
        match cmd_rx.try_recv() {
            Ok(CogCommand::Disconnect) | Ok(CogCommand::Shutdown) => {
                eprintln!("[cog] read: disconnect requested after {packet_count} packets");
                break;
            }
            _ => {}
        }

        // Sync: read byte-by-byte until we find the sync byte
        loop {
            stream.read_exact(&mut single).await?;
            if single[0] == SYNC_BYTE {
                packet_buf[0] = SYNC_BYTE;
                break;
            }
            sync_miss += 1;
        }

        // Read the remaining 194 bytes
        stream.read_exact(&mut packet_buf[1..]).await?;

        // Parse and send
        if let Some(sample) = parse_packet(&packet_buf) {
            packet_count += 1;
            if packet_count <= 3 || packet_count % 300 == 0 {
                eprintln!(
                    "[cog] read: pkt#{packet_count} counter={} status=0x{:02X} ch0={:.1}uV sync_misses={sync_miss}",
                    sample.counter, sample.status, sample.channels[0]
                );
            }
            // If the channel is full, drop the sample (backpressure)
            let _ = sample_tx.try_send(sample);
        }
    }

    eprintln!("[cog] connect: read loop ended, total packets={packet_count}");
    Ok(())
}

// ── macOS codepath ──────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
pub fn spawn_cog_worker() -> CogHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<CogCommand>();
    let (sample_tx, sample_rx) = mpsc::sync_channel::<CogSample>(1024);
    let (state_tx, state_rx) = mpsc::channel::<CogState>();

    std::thread::Builder::new()
        .name("cog-bt-worker".into())
        .spawn(move || {
            worker_loop_macos(cmd_rx, sample_tx, state_tx);
        })
        .expect("failed to spawn BT worker thread");

    CogHandle {
        cmd_tx,
        sample_rx,
        state_rx,
    }
}

#[cfg(target_os = "macos")]
fn worker_loop_macos(
    cmd_rx: mpsc::Receiver<CogCommand>,
    sample_tx: mpsc::SyncSender<CogSample>,
    state_tx: mpsc::Sender<CogState>,
) {
    loop {
        let cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break,
        };

        match cmd {
            CogCommand::StartScan => {
                send_state(&state_tx, CogState::Scanning);
                match scan_for_device_macos() {
                    Ok((port_path, name)) => {
                        send_state(
                            &state_tx,
                            CogState::Found {
                                id: port_path,
                                name,
                            },
                        );
                    }
                    Err(e) => {
                        send_state(&state_tx, CogState::Error(format!("Scan failed: {e}")));
                    }
                }
            }
            CogCommand::Connect(port) => {
                send_state(&state_tx, CogState::Connecting);
                match connect_and_stream_macos(&port, &cmd_rx, &sample_tx, &state_tx) {
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
                send_state(&state_tx, CogState::Disconnected);
            }
            CogCommand::Shutdown => break,
        }
    }
}

/// Scan for a Cognionics device by searching macOS virtual serial ports.
/// Paired Bluetooth SPP devices appear as `/dev/cu.<DeviceName>*`.
#[cfg(target_os = "macos")]
fn scan_for_device_macos() -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("[cog] scan: listing serial ports...");
    let ports = serialport::available_ports()?;

    for port in &ports {
        eprintln!("[cog] scan: port {:?}", port.port_name);
    }

    for port in &ports {
        let name = &port.port_name;
        let upper = name.to_uppercase();
        if upper.contains("HD-72") || upper.contains("HD72") || upper.contains("COGNIONICS") {
            eprintln!("[cog] scan: matched Cognionics port: {name}");
            // Use the filename portion as the display name
            let display = name
                .rsplit('/')
                .next()
                .unwrap_or(name)
                .strip_prefix("cu.")
                .unwrap_or(name);
            return Ok((name.clone(), display.to_string()));
        }
    }

    Err("No Cognionics serial port found — is the device paired?".into())
}

/// Open the serial port, send impedance-off, and enter a sync-byte-aligned read loop.
#[cfg(target_os = "macos")]
fn connect_and_stream_macos(
    port_path: &str,
    cmd_rx: &mpsc::Receiver<CogCommand>,
    sample_tx: &mpsc::SyncSender<CogSample>,
    state_tx: &mpsc::Sender<CogState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::{Read, Write};
    use std::time::Duration;

    eprintln!("[cog] connect: opening {port_path} at 1500000 baud...");
    let mut port = serialport::new(port_path, 1_500_000)
        .data_bits(serialport::DataBits::Eight)
        .parity(serialport::Parity::None)
        .stop_bits(serialport::StopBits::One)
        .timeout(Duration::from_millis(500))
        .open()?;

    eprintln!("[cog] connect: sending impedance-off (0x{IMPEDANCE_OFF_CMD:02X})...");
    port.write_all(&[IMPEDANCE_OFF_CMD])?;
    port.flush()?;
    eprintln!("[cog] connect: impedance-off sent, entering read loop");

    send_state(state_tx, CogState::Streaming);

    let mut packet_buf = [0u8; PACKET_SIZE];
    let mut single = [0u8; 1];
    let mut packet_count: u64 = 0;
    let mut sync_miss: u64 = 0;

    loop {
        // Check for disconnect command (non-blocking)
        match cmd_rx.try_recv() {
            Ok(CogCommand::Disconnect) | Ok(CogCommand::Shutdown) => {
                eprintln!("[cog] read: disconnect requested after {packet_count} packets");
                break;
            }
            _ => {}
        }

        // Sync: read byte-by-byte until we find the sync byte
        loop {
            match port.read_exact(&mut single) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    // Check for disconnect during timeout
                    match cmd_rx.try_recv() {
                        Ok(CogCommand::Disconnect) | Ok(CogCommand::Shutdown) => {
                            eprintln!(
                                "[cog] read: disconnect requested after {packet_count} packets"
                            );
                            return Ok(());
                        }
                        _ => continue,
                    }
                }
                Err(e) => return Err(e.into()),
            }
            if single[0] == SYNC_BYTE {
                packet_buf[0] = SYNC_BYTE;
                break;
            }
            sync_miss += 1;
        }

        // Read the remaining 194 bytes
        port.read_exact(&mut packet_buf[1..])?;

        // Parse and send
        if let Some(sample) = parse_packet(&packet_buf) {
            packet_count += 1;
            if packet_count <= 3 || packet_count % 300 == 0 {
                eprintln!(
                    "[cog] read: pkt#{packet_count} counter={} status=0x{:02X} ch0={:.1}uV sync_misses={sync_miss}",
                    sample.counter, sample.status, sample.channels[0]
                );
            }
            let _ = sample_tx.try_send(sample);
        }
    }

    eprintln!("[cog] connect: read loop ended, total packets={packet_count}");
    Ok(())
}

// ── Demo worker ──────────────────────────────────────────────────────────────

/// Spawn a synthetic-data demo worker that immediately enters Streaming state.
/// Generates EEG-like sine waves across all 64 channels without real hardware.
pub fn spawn_demo_worker() -> CogHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<CogCommand>();
    let (sample_tx, sample_rx) = mpsc::sync_channel::<CogSample>(1024);
    let (state_tx, state_rx) = mpsc::channel::<CogState>();

    std::thread::Builder::new()
        .name("cog-demo".into())
        .spawn(move || {
            demo_worker_loop(cmd_rx, sample_tx, state_tx);
        })
        .expect("failed to spawn demo worker thread");

    CogHandle {
        cmd_tx,
        sample_rx,
        state_rx,
    }
}

fn demo_worker_loop(
    cmd_rx: mpsc::Receiver<CogCommand>,
    sample_tx: mpsc::SyncSender<CogSample>,
    state_tx: mpsc::Sender<CogState>,
) {
    let _ = state_tx.send(CogState::Streaming);

    // Representative brain-wave frequencies across delta/theta/alpha/beta/gamma bands
    const BASE_FREQS: [f64; 8] = [1.5, 3.0, 6.0, 8.0, 10.0, 20.0, 35.0, 50.0];
    let mut phase = [0.0f64; NUM_CHANNELS];
    let mut counter: u8 = 0;
    let period = std::time::Duration::from_nanos(1_000_000_000 / 300);

    loop {
        match cmd_rx.try_recv() {
            Ok(CogCommand::Disconnect) | Ok(CogCommand::Shutdown) => {
                let _ = state_tx.send(CogState::Disconnected);
                break;
            }
            _ => {}
        }

        let mut channels = [0.0f32; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            let freq = BASE_FREQS[ch % BASE_FREQS.len()]
                * (1.0 + (ch / BASE_FREQS.len()) as f64 * 0.25);
            phase[ch] += 2.0 * std::f64::consts::PI * freq / 300.0;
            let offset = ((ch as f64 * 1.3 + counter as f64 * 0.02).sin()) * 8.0;
            channels[ch] = ((phase[ch].sin() * 45.0) + offset) as f32;
        }

        let _ = sample_tx.try_send(CogSample {
            counter,
            channels,
            status: VALID_STATUS,
        });
        counter = counter.wrapping_add(1);
        std::thread::sleep(period);
    }
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
