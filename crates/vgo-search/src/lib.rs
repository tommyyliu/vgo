#![forbid(unsafe_code)]

mod candidates;
mod coarse_fine;
mod evaluator;
mod mcts;
mod memory;
pub use memory::{HeapAllocation, TreeMemory};
#[cfg(feature = "iteration-lab")]
pub mod transition_lab;

pub use candidates::{Action, Candidate, CandidateSequence, CandidateSource, generate_candidates};
pub use coarse_fine::{CandidateSample, FineGrid, sample_candidates};
pub use evaluator::{Evaluation, EvaluationError, Evaluator, NaiveEvaluator, Policy};
mod stepped;
pub use mcts::{
    ChildSummary, SearchConfig, SearchResult, SearchStats, search, search_at_ply,
    search_with_evaluator,
};
pub use stepped::{
    ReusableSearch, ReuseMetrics, SteppedSearch, drive as drive_stepped, search_for_duration,
};
