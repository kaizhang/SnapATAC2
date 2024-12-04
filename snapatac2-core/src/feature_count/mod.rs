mod matrix;
mod counter;
mod data_iter;

pub use data_iter::{ChromValueIter, BaseData, FragmentData, ContactData, FragmentDataIter};
pub use counter::{FeatureCounter, CountingStrategy};
pub use matrix::{create_gene_matrix, create_tile_matrix, create_peak_matrix};