use core::f64;
use std::collections::HashMap;

const BC_MAX_QV: u8 = 66; // This is the illumina quality value
pub(crate) const BASE_OPTS: [u8; 4] = [b'A', b'C', b'G', b'T'];

pub struct Whitelist {
    whitelist_exists: bool,
    barcode_counts: HashMap<String, usize>,
    mismatch_count: usize,
    total_count: usize,
    total_base_count: u64,
    q30_base_count: u64,
    base_qual_sum: i64,
}

impl Whitelist {
    pub fn empty() -> Self {
        Self {
            whitelist_exists: false,
            barcode_counts: HashMap::new(),
            mismatch_count: 0,
            total_count: 0,
            total_base_count: 0,
            q30_base_count: 0,
            base_qual_sum: 0,
        }
    }

    pub fn new<I: IntoIterator<Item = S>, S: ToString>(iter: I) -> Self {
        let mut whitelist = Self::empty();
        whitelist.whitelist_exists = true;
        whitelist.barcode_counts = iter.into_iter().map(|x| (x.to_string(), 0)).collect();
        whitelist
    }

    /// Update the barcode counter with a barcode and its quality scores.
    pub fn count_barcode(&mut self, barcode: &str, barcode_qual: &[u8]) {
        if self.whitelist_exists {
            if let Some(count) = self.barcode_counts.get_mut(barcode) {
                *count += 1;
            } else {
                self.mismatch_count += 1;
            }
        } else if barcode.len() > 1 {
            *self.barcode_counts.entry(barcode.to_string()).or_insert(0) += 1;
        } else {
            self.mismatch_count += 1;
        }

        self.total_count += 1;

        for &qual in barcode_qual {
            let qual_int = (qual as u32) - 33;
            self.base_qual_sum += qual_int as i64;
            if qual_int >= 30 {
                self.q30_base_count += 1;
            }
            self.total_base_count += 1;
        }
    }

    pub fn get_barcode_counts(&self) -> &HashMap<String, usize> {
        &self.barcode_counts
    }

    pub fn get_mean_base_quality_score(&self) -> f64 {
        if self.total_base_count <= 0 {
            0.0
        } else {
            self.base_qual_sum as f64 / self.total_base_count as f64
        }
    }
}

/// A barcode validator that uses a barcode counter to validate barcodes.
pub struct BarcodeCorrector {
    /// threshold for sum of probability of error on barcode QVs. Barcodes exceeding
    /// this threshold will be marked as not valid.
    max_expected_errors: f64,
    /// if the posterior probability of a correction
    /// exceeds this threshold, the barcode will be corrected.
    bc_confidence_threshold: f64, 
}

impl Default for BarcodeCorrector {
    fn default() -> Self {
        Self {
            max_expected_errors: f64::MAX,
            bc_confidence_threshold: 0.975,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BarcodeError {
    ExceedExpectedError(f64),
    LowConfidence(f64),
    NoMatch,
}

impl BarcodeCorrector {
    /// Determine if a barcode is valid. A barcode is valid if any of the following conditions are met:
    /// 1) It is in the whitelist and the number of expected errors is less than the max_expected_errors.
    /// 2) It is not in the whitelist, but the number of expected errors is less than the max_expected_errors and the corrected barcode is in the whitelist.
    /// 3) If the whitelist does not exist, the barcode is always valid.
    /// 
    /// Return the corrected barcode
    pub fn correct(&self, barcode_counts: &HashMap<String, usize>, barcode: &str, qual: &[u8]) -> Result<String, BarcodeError> {
        let expected_errors: f64 = qual.iter().map(|&q| probability(q)).sum();
        if expected_errors >= self.max_expected_errors {
            return Err(BarcodeError::ExceedExpectedError(expected_errors));
        }
        if barcode_counts.contains_key(barcode) {
            return Ok(barcode.to_string());
        }

        let mut best_option = None;
        let mut total_likelihood = 0.0;
        let mut barcode_bytes = barcode.as_bytes().to_vec();
        for (pos, &qv) in qual.iter().enumerate() {
            let qv = qv.min(BC_MAX_QV);
            let existing = barcode_bytes[pos];
            for val in BASE_OPTS {
                if val != existing {
                    barcode_bytes[pos] = val;
                    let bc = std::str::from_utf8(&barcode_bytes).unwrap();
                    if let Some(raw_count) = barcode_counts.get(bc) {
                        let bc_count = 1 + raw_count;
                        let prob_edit = probability(qv);
                        let likelihood = bc_count as f64 * prob_edit;
                        match best_option {
                            None => best_option = Some((likelihood, bc.to_string())),
                            Some(ref old_best) => {
                                if old_best.0 < likelihood {
                                    best_option = Some((likelihood, bc.to_string()));
                                }
                            },
                        }
                        total_likelihood += likelihood;
                    }
                }
            }
            barcode_bytes[pos] = existing;
        }

        if let Some((best_like, best_bc)) = best_option {
            if best_like / total_likelihood >= self.bc_confidence_threshold {
                return Ok(best_bc)
            }
        }
        Err(BarcodeError::NoMatch)
    }
}

/// Convert quality scores to base-calling error probabilities.
fn probability(qual: u8) -> f64 {
    let offset = 33.0;
    10f64.powf(-((qual as f64 - offset) / 10.0))
}