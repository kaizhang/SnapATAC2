mod matrix;
mod coverage;

pub use coverage::{BaseData, FragmentData, ContactData, CountingStrategy, FragmentDataIter};
pub use matrix::{create_gene_matrix, create_tile_matrix, create_peak_matrix};