//! Diagnostic retained-payload accounting, not an allocator/RSS measurement.
use std::collections::HashSet;

/// Identity and reserved payload of a live allocation. Shared owners must report
/// the same identity. Allocator and reference-count headers are excluded.
#[derive(Clone, Copy, Debug)]
pub struct HeapAllocation {
    pub identity: usize,
    pub bytes: usize,
}

impl HeapAllocation {
    pub fn vector<T>(value: &Vec<T>) -> Self {
        Self {
            identity: value.as_ptr() as usize,
            bytes: value.capacity() * size_of::<T>(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TreeMemory {
    pub nodes: usize,
    pub node_bytes: usize,
    pub edge_capacity_bytes: usize,
    /// Live stone payload, excluding excess capacity hidden by Position's API.
    pub position_bytes: usize,
    pub policy_bytes: usize,
    pub sampling_bytes: usize,
    pub shared_bytes_avoided: usize,
    pub unreported_policies: usize,
    /// Legacy candidate caches are not accounted for in this first diagnostic.
    pub unreported_candidate_caches: usize,
}

impl TreeMemory {
    pub fn tracked_bytes(&self) -> usize {
        self.node_bytes
            + self.edge_capacity_bytes
            + self.position_bytes
            + self.policy_bytes
            + self.sampling_bytes
    }
}

#[derive(Default)]
pub(crate) struct MemoryCounter {
    pub report: TreeMemory,
    seen: HashSet<usize>,
}

impl MemoryCounter {
    pub fn allocation(&mut self, allocation: HeapAllocation, policy: bool) {
        if allocation.bytes == 0 {
            return;
        }
        if !self.seen.insert(allocation.identity) {
            self.report.shared_bytes_avoided += allocation.bytes;
        } else if policy {
            self.report.policy_bytes += allocation.bytes;
        } else {
            self.report.sampling_bytes += allocation.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_payload_is_counted_once_and_empty_vectors_are_free() {
        let values = vec![1.0f32; 128 * 128 + 1];
        let allocation = HeapAllocation::vector(&values);
        let mut counter = MemoryCounter::default();
        counter.allocation(allocation, true);
        counter.allocation(allocation, false);
        counter.allocation(HeapAllocation::vector(&Vec::<u8>::new()), false);
        assert_eq!(counter.report.policy_bytes, allocation.bytes);
        assert_eq!(counter.report.sampling_bytes, 0);
        assert_eq!(counter.report.shared_bytes_avoided, allocation.bytes);
        assert_eq!(counter.report.tracked_bytes(), allocation.bytes);
    }
}
