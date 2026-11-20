// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Mindcastle block-swap disk backend.
//!
//! A `.swap` image is a tiny text header plus a content-addressed
//! `swapdata-<uuid>/` directory (the on-disk format of the mindcastle
//! project; optionally with HTTP fallbacks and upload targets configured in
//! the header). This module wraps the out-of-tree `block-swap` crate, which
//! drives the C library behind `block-swap.c`.
//!
//! I/O goes through the C layer's asynchronous pipeline: reads and writes
//! complete on the backend's own threads (or synchronously in the submit
//! call), so the methods here block until the request is done. They are
//! meant to be called from a device worker thread, not from an event loop
//! that also serves other devices.
//!
//! Requests are byte-granular at this layer: extents wider than 512 bytes
//! are transparently expanded (read-modify-write) because the C layer works
//! in 512-byte sectors.
//!
//! The store has no trim concept, so [`SwapDisk::discard`] and
//! [`SwapDisk::write_zeroes`] publish the range as known-zero entries: it
//! reads back as zeroes without allocating chunks or writing data.

use std::io;
use std::path::Path;
use std::sync::Arc;

use block_swap::SECTOR_SIZE;
use block_swap::SwapImage;

/// A disk backed by a mindcastle `.swap` image.
///
/// The image is always open read-write: the format has no read-only mode,
/// since chunk data is shared and new writes land in new chunks.
#[derive(Debug)]
pub struct SwapDisk {
    image: Arc<SwapImage>,
}

impl SwapDisk {
    /// Opens the `.swap` header at `path`.
    pub fn open(path: &Path) -> io::Result<SwapDisk> {
        Ok(SwapDisk {
            image: Arc::new(SwapImage::open(path)?),
        })
    }

    /// Creates a new `.swap` image of `size_bytes` and opens it.
    pub fn create(path: &Path, size_bytes: u64) -> io::Result<SwapDisk> {
        SwapImage::create(path, size_bytes)?;
        Self::open(path)
    }

    /// Returns the image size in bytes, as recorded in the `.swap` header.
    pub fn len(&self) -> u64 {
        self.image.size_bytes()
    }

    /// Returns `true` if the image has no sectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads `buf.len()` bytes at `offset`.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let (extent_offset, extent_len, inner) = extent(offset, buf.len() as u64);
        if inner == 0 && extent_len == buf.len() as u64 {
            return self.image.read_at_blocking(offset, buf);
        }
        let mut bounce = vec![0u8; extent_len as usize];
        self.image.read_at_blocking(extent_offset, &mut bounce)?;
        let inner = inner as usize;
        buf.copy_from_slice(&bounce[inner..inner + buf.len()]);
        Ok(())
    }

    /// Writes `buf` at `offset`.
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let (extent_offset, extent_len, inner) = extent(offset, buf.len() as u64);
        if inner == 0 && extent_len == buf.len() as u64 {
            return self.image.write_at_blocking(offset, buf);
        }
        // Read-modify-write so that the bytes of the first/last sector that
        // the request does not cover are preserved.
        let mut bounce = vec![0u8; extent_len as usize];
        self.image.read_at_blocking(extent_offset, &mut bounce)?;
        let inner = inner as usize;
        bounce[inner..inner + buf.len()].copy_from_slice(buf);
        self.image.write_at_blocking(extent_offset, &bounce)
    }

    /// Flushes everything written so far, including uploads when the header
    /// configures them.
    pub fn flush(&self) -> io::Result<()> {
        self.image.flush_blocking()
    }

    /// Discards `len` bytes at `offset`.
    ///
    /// The store has no trim, so the range is published as zeroes.
    pub fn discard(&self, offset: u64, len: u64) -> io::Result<()> {
        self.write_zeroes(offset, len)
    }

    /// Writes zeroes over `len` bytes at `offset`.
    ///
    /// The engine's trim is sector-granular; a byte-unaligned head or tail
    /// is handled here so that the bytes of boundary sectors that the
    /// request does not cover survive.
    pub fn write_zeroes(&self, offset: u64, len: u64) -> io::Result<()> {
        let end = offset.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "zero-write range overflows")
        })?;
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero-write beyond end of image",
            ));
        }
        if len == 0 {
            return Ok(());
        }

        let aligned_start = offset.saturating_add(SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1);
        let aligned_end = end & !(SECTOR_SIZE - 1);
        if aligned_start >= aligned_end {
            return self.zero_partial(offset, len);
        }
        if offset < aligned_start {
            self.zero_partial(offset, aligned_start - offset)?;
        }
        self.image
            .trim_blocking(aligned_start, aligned_end - aligned_start)?;
        if aligned_end < end {
            self.zero_partial(aligned_end, end - aligned_end)?;
        }
        Ok(())
    }

    /// Zeroes a sub-sector range by reading and rewriting the covering
    /// sectors. Only used for the byte-unaligned ends of a zero write.
    fn zero_partial(&self, offset: u64, len: u64) -> io::Result<()> {
        let (extent_offset, extent_len, inner) = extent(offset, len);
        let mut buf = vec![0u8; extent_len as usize];
        self.image.read_at_blocking(extent_offset, &mut buf)?;
        let inner = inner as usize;
        buf[inner..inner + len as usize].fill(0);
        self.image.write_at_blocking(extent_offset, &buf)
    }
}

/// Expands an arbitrary byte range to the covering 512-byte sector extent.
///
/// Returns `(extent_offset, extent_len, offset_of_request_within_extent)`.
fn extent(offset: u64, len: u64) -> (u64, u64, u64) {
    let extent_offset = offset & !(SECTOR_SIZE - 1);
    let extent_end = offset
        .checked_add(len)
        .map(|end| (end + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1));
    let extent_len = extent_end.unwrap_or(0) - extent_offset;
    (extent_offset, extent_len, offset - extent_offset)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    const IMAGE_SIZE: u64 = 8 << 20; // 8 MiB: several zero chunks

    fn new_image() -> (TempDir, SwapDisk) {
        let dir = TempDir::new().unwrap();
        let disk = SwapDisk::create(&dir.path().join("test.swap"), IMAGE_SIZE).unwrap();
        (dir, disk)
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed) | 1)
            .collect()
    }

    /// Fills the whole image with a non-zero, non-repeating pattern.
    fn seed(disk: &SwapDisk) -> Vec<u8> {
        let data = pattern(IMAGE_SIZE as usize, 7);
        disk.write_at(0, &data).unwrap();
        data
    }

    fn read_all(disk: &SwapDisk) -> Vec<u8> {
        let mut buf = vec![0u8; IMAGE_SIZE as usize];
        disk.read_at(0, &mut buf).unwrap();
        buf
    }

    #[test]
    fn extent_math() {
        assert_eq!(extent(0, 512), (0, 512, 0));
        assert_eq!(extent(512, 512), (512, 512, 0));
        assert_eq!(extent(200, 100), (0, 512, 200));
        assert_eq!(extent(1 << 20, 4096), (1 << 20, 4096, 0));
        assert_eq!(extent(1000, 4096), (512, 4096 + 512, 488));
    }

    #[test]
    fn unaligned_write_preserves_neighbours() {
        let (_dir, disk) = new_image();
        let before = seed(&disk);
        let data = pattern(1000, 3);
        disk.write_at(777, &data).unwrap();
        let after = read_all(&disk);
        assert_eq!(&after[..777], &before[..777]);
        assert_eq!(&after[777..777 + data.len()], &data[..]);
        assert_eq!(&after[777 + data.len()..], &before[777 + data.len()..]);
    }

    #[test]
    fn discard_zeroes_only_the_requested_range() {
        let (_dir, disk) = new_image();
        let before = seed(&disk);
        // Unaligned at both ends and spanning more than one zero chunk.
        let offset = 1000usize;
        let len = (2 << 20) + 123;
        disk.discard(offset as u64, len as u64).unwrap();
        let after = read_all(&disk);
        assert_eq!(&after[..offset], &before[..offset]);
        assert!(after[offset..offset + len].iter().all(|&b| b == 0));
        assert_eq!(&after[offset + len..], &before[offset + len..]);
    }

    #[test]
    fn write_zeroes_covers_the_whole_device() {
        let (_dir, disk) = new_image();
        seed(&disk);
        disk.write_zeroes(0, IMAGE_SIZE).unwrap();
        assert!(read_all(&disk).iter().all(|&b| b == 0));
    }

    #[test]
    fn write_zeroes_aligned_chunks() {
        let (_dir, disk) = new_image();
        let before = seed(&disk);
        let offset = 2 << 20;
        let len = 3 << 20;
        disk.write_zeroes(offset, len).unwrap();
        let after = read_all(&disk);
        assert_eq!(&after[..offset as usize], &before[..offset as usize]);
        assert!(
            after[offset as usize..(offset + len) as usize]
                .iter()
                .all(|&b| b == 0)
        );
        assert_eq!(
            &after[(offset + len) as usize..],
            &before[(offset + len) as usize..]
        );
    }

    #[test]
    fn write_after_discard_is_not_lost() {
        // Regression test for the C library's insert/trim race: writing,
        // discarding the whole device and writing again used to abort in
        // swap_insert_thread (block-swap.c:557) at close.
        let (_dir, disk) = new_image();
        seed(&disk);
        disk.discard(0, IMAGE_SIZE).unwrap();
        assert!(read_all(&disk).iter().all(|&b| b == 0));
        let data = pattern(IMAGE_SIZE as usize, 11);
        disk.write_at(0, &data).unwrap();
        disk.flush().unwrap();
        assert_eq!(read_all(&disk), data);
    }

    #[test]
    fn zero_length_is_a_noop() {
        let (_dir, disk) = new_image();
        let before = seed(&disk);
        disk.write_zeroes(512, 0).unwrap();
        assert_eq!(read_all(&disk), before);
    }

    #[test]
    fn zero_write_beyond_end_fails() {
        let (_dir, disk) = new_image();
        assert!(disk.write_zeroes(IMAGE_SIZE - 512, 1024).is_err());
    }
}
