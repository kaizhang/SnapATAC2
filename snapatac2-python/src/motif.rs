use pyo3::prelude::*;
use numpy::{Ix2, PyReadonlyArray};
use std::fs::File;
use std::path::Path;
use std::io::Read;

use snapatac2_core::{
    motif,
};

#[pyclass]
#[repr(transparent)]
pub struct PyDNAMotif(pub motif::DNAMotif);

#[pymethods]
impl PyDNAMotif {
    #[new]
    fn new<'py>(
        name: &str,
        matrix: &'py PyAny,
    ) -> Self {
        let pwm: PyReadonlyArray<f64, Ix2> = matrix.extract().unwrap();
        let motif = motif::DNAMotif {
            name: name.to_string(),
            probability: pwm.as_array().rows().into_iter()
                .map(|row| row.into_iter().map(|x| *x).collect::<Vec<_>>().try_into().unwrap()).collect(),
        };
        PyDNAMotif(motif)
    }

    fn with_background(&self, a: f64, c: f64, g: f64, t: f64) -> PyDNAMotifScanner {
        PyDNAMotifScanner(self.0.clone().to_scanner(motif::BackgroundProb([a, c, g, t])))
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyDNAMotifScanner(pub motif::DNAMotifScanner);

#[pymethods]
impl PyDNAMotifScanner {
    fn find(&self, seq: &str, pvalue: f64) -> Vec<(usize, f64)> {
        self.0.find(seq.as_bytes(), pvalue).collect()
    }

    fn exist(&self, seq: &str, pvalue: f64) -> bool {
        self.0.find(seq.as_bytes(), pvalue).next().is_some()
    }
}

#[pyfunction]
pub(crate) fn read_motifs<'py>(
    py: Python<'py>,
    filename: &str,
) -> Vec<PyDNAMotif> {
    let path = Path::new(filename);
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open file: {}", why),
        Ok(file) => file,
    };
    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();
    motif::parse_meme(&s).into_iter().map(|x| PyDNAMotif(x)).collect()
}
 