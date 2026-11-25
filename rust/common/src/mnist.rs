//! Pure-Rust MNIST IDX loader (byte-for-byte compatible with `torchvision.datasets.MNIST`).
//!
//! Search order for raw IDX files:
//! 1. `MNIST_DIR` environment variable (if set)
//! 2. `$HOME/.cache/modernaicourse/mnist`
//! 3. `$HOME/modernaicourse/homework/hw1/MNIST/raw` (existing Python course download, if present)
//!
//! If not found locally, downloads the standard `.gz` files from the PyTorch MNIST mirror
//! into `$HOME/.cache/modernaicourse/mnist` and decompresses them automatically.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use candle_core::{Device, Error, Result, Tensor};
use flate2::read::GzDecoder;

const MIRROR_BASE: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

/// A loaded MNIST split with normalized grayscale pixel values in `[0.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct MnistSplit {
    /// `f32` tensor of shape `[N, 28, 28]` with pixel values in `[0.0, 1.0]`.
    pub images: Tensor,
    /// Digit labels (`0..=9`), length `N`.
    pub labels: Vec<u8>,
}

impl MnistSplit {
    /// Number of images in this split.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split is empty.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Return the `i`-th image as a `[28, 28]` `f32` tensor.
    pub fn image(&self, i: usize) -> Result<Tensor> {
        self.images.get(i)
    }
}

fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MNIST_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/modernaicourse/mnist")
}

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("MNIST_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".cache/modernaicourse/mnist"));
        dirs.push(home.join("modernaicourse/homework/hw1/MNIST/raw"));
    }
    dirs
}

fn ensure_file(filename: &str) -> Result<PathBuf> {
    for dir in candidate_dirs() {
        let candidate = dir.join(filename);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    let dir = cache_dir();
    fs::create_dir_all(&dir).map_err(Error::wrap)?;
    let dest = dir.join(filename);
    if dest.is_file() {
        return Ok(dest);
    }

    let url = format!("{MIRROR_BASE}/{filename}.gz");
    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| Error::Msg(format!("failed to download {url}: {e}")))?;
    let mut decoder = GzDecoder::new(response.body_mut().as_reader());
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Msg(format!("failed to decompress {url}: {e}")))?;

    // Atomic-ish write via temp file to avoid partial downloads under parallel tests.
    let tmp = dir.join(format!("{filename}.tmp.{}", std::process::id()));
    fs::write(&tmp, &bytes).map_err(Error::wrap)?;
    fs::rename(&tmp, &dest).map_err(Error::wrap)?;
    Ok(dest)
}

fn read_u32_be<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_idx_images(path: &Path) -> Result<(usize, usize, usize, Vec<u8>)> {
    let mut f = File::open(path).map_err(Error::wrap)?;
    let magic = read_u32_be(&mut f).map_err(Error::wrap)?;
    if magic != 2051 {
        candle_core::bail!("invalid MNIST image magic number {magic} in {}", path.display());
    }
    let count = read_u32_be(&mut f).map_err(Error::wrap)? as usize;
    let rows = read_u32_be(&mut f).map_err(Error::wrap)? as usize;
    let cols = read_u32_be(&mut f).map_err(Error::wrap)? as usize;
    let mut data = vec![0u8; count * rows * cols];
    f.read_exact(&mut data).map_err(Error::wrap)?;
    Ok((count, rows, cols, data))
}

fn read_idx_labels(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(Error::wrap)?;
    let magic = read_u32_be(&mut f).map_err(Error::wrap)?;
    if magic != 2049 {
        candle_core::bail!("invalid MNIST label magic number {magic} in {}", path.display());
    }
    let count = read_u32_be(&mut f).map_err(Error::wrap)? as usize;
    let mut labels = vec![0u8; count];
    f.read_exact(&mut labels).map_err(Error::wrap)?;
    Ok(labels)
}

fn load_split(images_file: &str, labels_file: &str, device: &Device) -> Result<MnistSplit> {
    let img_path = ensure_file(images_file)?;
    let lbl_path = ensure_file(labels_file)?;

    let (count, rows, cols, raw_pixels) = read_idx_images(&img_path)?;
    let labels = read_idx_labels(&lbl_path)?;
    if labels.len() != count {
        candle_core::bail!(
            "MNIST count mismatch: {} images vs {} labels",
            count,
            labels.len()
        );
    }

    let pixels: Vec<f32> = raw_pixels.into_iter().map(|b| b as f32 / 255.0).collect();
    let images = Tensor::from_vec(pixels, (count, rows, cols), device)?;
    Ok(MnistSplit { images, labels })
}

/// Load the full 60,000-image MNIST training set (`images`: `[60000, 28, 28]` `f32` in `[0, 1]`).
pub fn load_train(device: &Device) -> Result<MnistSplit> {
    load_split(TRAIN_IMAGES, TRAIN_LABELS, device)
}

/// Load the full 10,000-image MNIST test set (`images`: `[10000, 28, 28]` `f32` in `[0, 1]`).
pub fn load_test(device: &Device) -> Result<MnistSplit> {
    load_split(TEST_IMAGES, TEST_LABELS, device)
}

/// Load only the `0` and `1` digits from the MNIST training set, preserving exact MNIST order.
///
/// Matches the filtered `images` dataset in `homework/hw1/hw1_tests.py`.
pub fn load_train_zero_one(device: &Device) -> Result<MnistSplit> {
    let img_path = ensure_file(TRAIN_IMAGES)?;
    let lbl_path = ensure_file(TRAIN_LABELS)?;

    let (_count, rows, cols, raw_pixels) = read_idx_images(&img_path)?;
    let all_labels = read_idx_labels(&lbl_path)?;
    let stride = rows * cols;

    let mut filtered_pixels = Vec::new();
    let mut filtered_labels = Vec::new();
    for (idx, &label) in all_labels.iter().enumerate() {
        if label == 0 || label == 1 {
            let start = idx * stride;
            filtered_pixels.extend(
                raw_pixels[start..start + stride]
                    .iter()
                    .map(|&b| b as f32 / 255.0),
            );
            filtered_labels.push(label);
        }
    }

    let n = filtered_labels.len();
    let images = Tensor::from_vec(filtered_pixels, (n, rows, cols), device)?;
    Ok(MnistSplit {
        images,
        labels: filtered_labels,
    })
}
