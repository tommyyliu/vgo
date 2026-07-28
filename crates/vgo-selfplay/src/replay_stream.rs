#![forbid(unsafe_code)]

use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use vgo_raster::{CHANNEL_COUNT, RasterConfig, SemanticRaster};

pub(crate) const REPLAY_MAGIC: [u8; 8] = *b"VGORPLY1";
// v2 added raw per-cell visit counts and coarse->fine sampling probability
// (beta). v3 additionally records the empirical proposal multiplicity per cell.
pub(crate) const REPLAY_VERSION: u32 = 3;

pub(crate) struct LabeledSample {
    pub(crate) raster: SemanticRaster,
    pub(crate) policy: Vec<f32>,
    pub(crate) policy_mask: Vec<f32>,
    pub(crate) visits: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) proposal_counts: Vec<u32>,
    pub(crate) value: f32,
    pub(crate) selected_action: u32,
    pub(crate) game: u64,
    pub(crate) ply: u32,
    pub(crate) seed: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GameWrite {
    pub(crate) samples_written: usize,
    pub(crate) samples_truncated: usize,
}

#[derive(Debug)]
pub(crate) struct PublishedReplay {
    pub(crate) samples: usize,
    pub(crate) sha256: String,
    pub(crate) bytes: u64,
    pub(crate) examples: Vec<SemanticRaster>,
    pub(crate) first_game_id: Option<u64>,
    pub(crate) last_game_id: Option<u64>,
    pub(crate) write_time: Duration,
    pub(crate) sync_time: Duration,
}

/// Incrementally writes replay-v3 records while games are still being played.
///
/// Only complete, terminally-labelled games enter this writer. The final game
/// may be cut at the configured sample boundary, but no partial record is ever
/// published. The advertised sample count is therefore known before the first
/// record is written and remains compatible with the existing memory-mapped v3
/// loader.
pub(crate) struct ReplayStream {
    final_path: PathBuf,
    temporary_path: PathBuf,
    writer: Option<BufWriter<HashingWriter>>,
    raster: RasterConfig,
    policy_size: usize,
    target_samples: usize,
    samples_written: usize,
    examples_limit: usize,
    examples: Vec<SemanticRaster>,
    first_game_id: Option<u64>,
    last_game_id: Option<u64>,
    write_time: Duration,
    published: bool,
}

impl ReplayStream {
    pub(crate) fn create(
        path: &Path,
        target_samples: usize,
        raster: RasterConfig,
        policy_size: usize,
        examples_limit: usize,
    ) -> io::Result<Self> {
        let samples_u32 = u32::try_from(target_samples).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "replay sample count does not fit the v3 header",
            )
        })?;
        let channels_u32 = u32::try_from(CHANNEL_COUNT).expect("channel count fits in u32");
        let height_u32 = u32::try_from(raster.height).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "raster height does not fit u32",
            )
        })?;
        let width_u32 = u32::try_from(raster.width).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "raster width does not fit u32")
        })?;
        let policy_u32 = u32::try_from(policy_size).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "policy size does not fit u32")
        })?;
        if target_samples == 0 || raster.width == 0 || raster.height == 0 || policy_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "replay dimensions and sample count must be positive",
            ));
        }
        let temporary_path = temporary_path(path);
        if path.exists() || temporary_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("replay output already exists: {}", path.display()),
            ));
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, HashingWriter::new(file));
        writer.write_all(&REPLAY_MAGIC)?;
        for value in [
            REPLAY_VERSION,
            samples_u32,
            channels_u32,
            height_u32,
            width_u32,
            policy_u32,
        ] {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(Self {
            final_path: path.to_path_buf(),
            temporary_path,
            writer: Some(writer),
            raster,
            policy_size,
            target_samples,
            samples_written: 0,
            examples_limit,
            examples: Vec::with_capacity(examples_limit.min(target_samples)),
            first_game_id: None,
            last_game_id: None,
            write_time: Duration::ZERO,
            published: false,
        })
    }

    pub(crate) const fn is_full(&self) -> bool {
        self.samples_written == self.target_samples
    }

    pub(crate) fn write_game(&mut self, samples: Vec<LabeledSample>) -> io::Result<GameWrite> {
        let remaining = self.target_samples - self.samples_written;
        let to_write = remaining.min(samples.len());
        let truncated = samples.len() - to_write;
        let started = Instant::now();
        for sample in samples.into_iter().take(to_write) {
            self.validate_sample(&sample)?;
            if self.examples.len() < self.examples_limit {
                self.examples.push(sample.raster.clone());
            }
            self.first_game_id.get_or_insert(sample.game);
            self.last_game_id = Some(sample.game);
            write_sample(
                self.writer
                    .as_mut()
                    .expect("writer exists until publication"),
                &sample,
            )?;
            self.samples_written += 1;
        }
        self.write_time += started.elapsed();
        Ok(GameWrite {
            samples_written: to_write,
            samples_truncated: truncated,
        })
    }

    pub(crate) fn publish(mut self) -> io::Result<PublishedReplay> {
        if !self.is_full() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "cannot publish replay with {} of {} samples",
                    self.samples_written, self.target_samples
                ),
            ));
        }
        let sync_started = Instant::now();
        let mut writer = self.writer.take().expect("writer exists until publication");
        writer.flush()?;
        let hashing = writer.into_inner().map_err(|error| error.into_error())?;
        let (file, digest, bytes) = hashing.into_parts();
        file.sync_all()?;
        drop(file);
        if self.final_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "replay output appeared during generation: {}",
                    self.final_path.display()
                ),
            ));
        }
        fs::rename(&self.temporary_path, &self.final_path)?;
        sync_parent_directory(&self.final_path)?;
        self.published = true;
        Ok(PublishedReplay {
            samples: self.samples_written,
            sha256: format!("{:x}", digest.finalize()),
            bytes,
            examples: std::mem::take(&mut self.examples),
            first_game_id: self.first_game_id,
            last_game_id: self.last_game_id,
            write_time: self.write_time,
            sync_time: sync_started.elapsed(),
        })
    }

    fn validate_sample(&self, sample: &LabeledSample) -> io::Result<()> {
        let dimensions_match = sample.raster.config() == self.raster;
        let policies_match = [
            sample.policy.len(),
            sample.policy_mask.len(),
            sample.visits.len(),
            sample.beta.len(),
            sample.proposal_counts.len(),
        ]
        .into_iter()
        .all(|length| length == self.policy_size);
        if !dimensions_match || !policies_match {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sample dimensions do not match the replay header",
            ));
        }
        if sample.selected_action as usize >= self.policy_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "selected action is outside the replay policy",
            ));
        }
        Ok(())
    }
}

impl Drop for ReplayStream {
    fn drop(&mut self) {
        if !self.published {
            // This file is private staging state created by this writer. Removing
            // it makes a failed attempt safely retryable and never touches a
            // published replay.
            self.writer.take();
            let _ = fs::remove_file(&self.temporary_path);
        }
    }
}

struct HashingWriter {
    file: File,
    digest: Sha256,
    bytes: u64,
}

impl HashingWriter {
    fn new(file: File) -> Self {
        Self {
            file,
            digest: Sha256::new(),
            bytes: 0,
        }
    }

    fn into_parts(self) -> (File, Sha256, u64) {
        (self.file, self.digest, self.bytes)
    }
}

impl Write for HashingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.file.write(buffer)?;
        self.digest.update(&buffer[..written]);
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn write_f32(writer: &mut impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_sample(writer: &mut impl Write, sample: &LabeledSample) -> io::Result<()> {
    for &value in sample.raster.data() {
        write_f32(writer, value)?;
    }
    for values in [
        &sample.policy,
        &sample.policy_mask,
        &sample.visits,
        &sample.beta,
    ] {
        for &value in values {
            write_f32(writer, value)?;
        }
    }
    for &value in &sample.proposal_counts {
        writer.write_all(&value.to_le_bytes())?;
    }
    write_f32(writer, sample.value)?;
    writer.write_all(&sample.selected_action.to_le_bytes())?;
    writer.write_all(&sample.game.to_le_bytes())?;
    writer.write_all(&sample.ply.to_le_bytes())?;
    writer.write_all(&sample.seed.to_le_bytes())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut path = path.as_os_str().to_os_string();
    path.push(".tmp");
    PathBuf::from(path)
}

pub(crate) fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Read,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use sha2::{Digest, Sha256};
    use vgo_core::{Color, Position};
    use vgo_raster::{CHANNEL_COUNT, RasterConfig, rasterize};

    use super::{LabeledSample, REPLAY_MAGIC, REPLAY_VERSION, ReplayStream, temporary_path};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create() -> Self {
            let serial = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("vgo-replay-stream-{}-{serial}", std::process::id()));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sample(game: u64, ply: u32) -> LabeledSample {
        let config = RasterConfig::square(2);
        let policy_size = config.pixels() + 1;
        LabeledSample {
            raster: rasterize(&Position::new(1.0 / 6.0, Vec::new(), Color::Black), config),
            policy: vec![0.2; policy_size],
            policy_mask: vec![1.0; policy_size],
            visits: vec![1.0; policy_size],
            beta: vec![0.0; policy_size],
            proposal_counts: vec![0; policy_size],
            value: 1.0,
            selected_action: 0,
            game,
            ply,
            seed: 50_001 + game,
        }
    }

    #[test]
    fn streams_exactly_the_advertised_v3_records_and_hashes_them() {
        let directory = TestDirectory::create();
        let path = directory.0.join("dataset.vgo");
        let raster = RasterConfig::square(2);
        let policy_size = raster.pixels() + 1;
        let mut stream =
            ReplayStream::create(&path, 3, raster, policy_size, 2).expect("create replay");

        assert_eq!(
            stream
                .write_game(vec![sample(7, 0), sample(7, 1)])
                .expect("write first game")
                .samples_truncated,
            0
        );
        let final_game = stream
            .write_game(vec![sample(9, 0), sample(9, 1)])
            .expect("write final game");
        assert_eq!(final_game.samples_written, 1);
        assert_eq!(final_game.samples_truncated, 1);

        let published = stream.publish().expect("publish replay");
        assert_eq!(published.samples, 3);
        assert_eq!(published.examples.len(), 2);
        assert_eq!(published.first_game_id, Some(7));
        assert_eq!(published.last_game_id, Some(9));
        assert!(!temporary_path(&path).exists());

        let bytes = fs::read(&path).expect("read replay");
        let record_bytes =
            (CHANNEL_COUNT * raster.pixels() + 5 * policy_size) * size_of::<f32>() + 28;
        assert_eq!(bytes.len(), 32 + 3 * record_bytes);
        assert_eq!(&bytes[..8], &REPLAY_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            REPLAY_VERSION
        );
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        assert_eq!(published.bytes, bytes.len() as u64);
        assert_eq!(published.sha256, format!("{:x}", Sha256::digest(&bytes)));
    }

    #[test]
    fn incomplete_or_failed_streams_never_publish() {
        let directory = TestDirectory::create();
        let path = directory.0.join("dataset.vgo");
        let raster = RasterConfig::square(2);
        let policy_size = raster.pixels() + 1;
        let mut stream =
            ReplayStream::create(&path, 2, raster, policy_size, 0).expect("create replay");
        stream
            .write_game(vec![sample(1, 0)])
            .expect("write partial replay");
        assert_eq!(
            stream
                .publish()
                .expect_err("partial replay cannot publish")
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        assert!(!path.exists());
        assert!(!temporary_path(&path).exists());

        File::create(&path).expect("reserve published path");
        let error = match ReplayStream::create(&path, 1, raster, policy_size, 0) {
            Ok(_) => panic!("published path cannot be overwritten"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn rolling_digest_covers_bytes_that_reach_the_file() {
        let directory = TestDirectory::create();
        let path = directory.0.join("dataset.vgo");
        let raster = RasterConfig::square(2);
        let policy_size = raster.pixels() + 1;
        let mut stream =
            ReplayStream::create(&path, 1, raster, policy_size, 0).expect("create replay");
        stream.write_game(vec![sample(4, 0)]).expect("write replay");
        let published = stream.publish().expect("publish replay");

        let mut reader = File::open(path).expect("open replay");
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 37];
        loop {
            let count = reader.read(&mut buffer).expect("read replay");
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        assert_eq!(published.sha256, format!("{:x}", digest.finalize()));
    }
}
