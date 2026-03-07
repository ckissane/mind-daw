use lsl::Pullable;
use std::collections::VecDeque;

/// Metadata about a discovered LSL stream.
#[derive(Clone, Debug)]
pub struct StreamMeta {
    pub name: String,
    pub stream_type: String,
    pub channel_count: i32,
    pub sample_rate: f64,
    pub source_id: String,
}

/// Discover available LSL streams on the network.
///
/// Blocks for `wait_secs` while resolving streams.
pub fn discover_streams(wait_secs: f64) -> Vec<StreamMeta> {
    match lsl::resolve_streams(wait_secs) {
        Ok(results) => results
            .iter()
            .map(|info| StreamMeta {
                name: info.stream_name(),
                stream_type: info.stream_type(),
                channel_count: info.channel_count(),
                sample_rate: info.nominal_srate(),
                source_id: info.source_id(),
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// An active connection to a paired LSL stream.
pub struct PairedStream {
    pub meta: StreamMeta,
    inlet: lsl::StreamInlet,
    /// Ring buffer of recent samples per channel: buffer[channel][sample]
    pub buffer: Vec<VecDeque<f32>>,
    pub buffer_capacity: usize,
}

impl PairedStream {
    /// Connect to a specific stream by re-resolving it and opening an inlet.
    ///
    /// This blocks for up to ~5s during resolve + open. Call from the main
    /// thread (GPUI executor) since StreamInlet is !Send.
    pub fn connect(meta: &StreamMeta, buffer_capacity: usize) -> anyhow::Result<Self> {
        let pred = format!("name='{}' and type='{}'", meta.name, meta.stream_type);
        let results = lsl::resolve_bypred(&pred, 1, 5.0)?;
        let info = results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Stream '{}' not found", meta.name))?;

        let channel_count = meta.channel_count as usize;
        let inlet = lsl::StreamInlet::new(&info, 360, 0, true)?;
        inlet.open_stream(5.0)?;

        let buffer = vec![VecDeque::with_capacity(buffer_capacity); channel_count];

        Ok(Self {
            meta: meta.clone(),
            inlet,
            buffer,
            buffer_capacity,
        })
    }

    /// Pull available samples from the inlet into the ring buffer.
    /// Returns the number of new samples pulled.
    pub fn pull_samples(&mut self) -> usize {
        let channel_count = self.meta.channel_count as usize;
        let mut count = 0;

        // Pull f32 samples with timeout=0 (non-blocking)
        loop {
            match Pullable::<f32>::pull_sample(&self.inlet, 0.0) {
                Ok((sample, timestamp)) if timestamp > 0.0 => {
                    for (ch, &val) in sample.iter().enumerate().take(channel_count) {
                        let buf = &mut self.buffer[ch];
                        if buf.len() >= self.buffer_capacity {
                            buf.pop_front();
                        }
                        buf.push_back(val);
                    }
                    count += 1;
                }
                _ => break,
            }
        }

        count
    }

    /// Get the latest buffer contents for a given channel.
    pub fn channel_data(&self, channel: usize) -> Vec<f32> {
        self.buffer
            .get(channel)
            .map(|buf| buf.iter().copied().collect())
            .unwrap_or_default()
    }
}
