pub mod qc;
pub mod matrix;
pub mod mark_duplicates;
pub mod fragment;
pub mod genome;
pub mod counter;

pub use fragment::{make_fragment_file, import_fragments};
pub use genome::{SnapData, Promoters, Transcript, read_transcripts};
pub use mark_duplicates::FlagStat;
pub use qc::{Fragment, CellBarcode, read_tss, make_promoter_map, get_barcode_count};
pub use matrix::{create_gene_matrix, create_tile_matrix, create_peak_matrix};