pub mod qc;
pub mod bam;
pub mod count_data;
pub mod barcode;

pub use count_data::{import_fragments, import_contacts, Promoters, Transcript,
    read_transcripts_from_gff, read_transcripts_from_gtf,
    create_gene_matrix, create_tile_matrix, create_peak_matrix,
    GenomeCount, ContactMap, SnapData,
};
pub use bam::{make_fragment_file, FlagStat, BamQC};
pub use qc::{Fragment, Contact, CellBarcode, read_tss, TssRegions, make_promoter_map, get_barcode_count};