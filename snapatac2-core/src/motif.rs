use itertools::Itertools;
use std::{io::{Error, ErrorKind}, str::FromStr, default::Default};

#[derive(Debug, Clone)]
pub struct BackgroundProb(pub [f64; 4]);

impl Default for BackgroundProb {
    fn default() -> Self {
        BackgroundProb([0.25, 0.25, 0.25, 0.25])
    }
}

#[derive(Debug, Clone)]
pub struct DNAMotif {
    pub id: String,
    pub name: Option<String>,
    pub family: Option<String>,
    pub probability: Vec<[f64; 4]>,
}

impl FromStr for DNAMotif {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut lines = s.lines();
        let first_line = lines.next().unwrap();
        if &first_line[..5] != "MOTIF" {
            return Err(Error::new(ErrorKind::Other, "MOTIF not found"));
        }
        // skip
        lines.next();
        let pwm = lines.map(|x| x.trim().split_ascii_whitespace()
            .map(|v| v.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        ).collect();

        Ok(DNAMotif {
            id: first_line.strip_prefix("MOTIF ").unwrap().to_string(),
            name: None,
            family: None,
            probability: pwm,
        })
    }
}

impl DNAMotif {
    pub fn size(&self) -> usize { self.probability.len() }

    pub fn info_content(&self) -> f64 {
        self.probability.iter().map(|row| {
            let entropy: f64 = row.into_iter().map(|p| if *p == 0.0 {
                0.0
            } else {
                -1.0 * *p * p.log2()
            }).sum();
            2.0 - entropy
        }).sum()
    }

    pub fn to_scanner(mut self, bg: BackgroundProb) -> DNAMotifScanner {
        self.add_pseudocount(0.0001);
        let cdf = ScoreCDF::new(&self, &bg);
        DNAMotifScanner {
            motif: self,
            cdf,
            background: bg,
        }
    }

    pub fn revcomp(&self) -> Self {
        todo!()
    }

    fn add_pseudocount(&mut self, pseudocount: f64) {
        self.probability.iter_mut().for_each(|ps| {
            ps.iter_mut().for_each(|p| if *p == 0.0 { *p = pseudocount; });
            let s: f64 = ps.iter().sum();
            if s != 1.0 {
                ps.iter_mut().for_each(|p| *p /= s);
            }
        });
    }

    fn optimal_scores_suffix(&self, bg: &BackgroundProb) -> Vec<f64> {
        let mut scores: Vec<f64> = self.probability.iter().scan(0.0, |state, prob| {
            let (i, p) = prob.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            *state = *state + (p / bg.0[i]).ln();
            Some(*state)
        }).collect();
        let max = *scores.last().unwrap();
        scores.iter_mut().for_each(|x| *x = max - *x);
        scores
    }

    // This function does not do bound checks on seq.
    fn look_ahead_search(
        &self,
        bg: &BackgroundProb,
        remain_best: &Vec<f64>,  // best possible match score of suffixes
        seq: &[u8],
        start: usize,
        thres: f64,
    ) -> Option<(usize, f64)> {
        let n = self.size();
        let mut cur_pos = 0;
        let mut cur_match = 0.0;
        loop {
            let sc = match seq[cur_pos + start] {
                b'A' | b'a' => (self.probability[cur_pos][0] / bg.0[0]).ln(),
                b'C' | b'c' => (self.probability[cur_pos][1] / bg.0[1]).ln(),
                b'G' | b'g' => (self.probability[cur_pos][2] / bg.0[2]).ln(),
                b'T' | b't' => (self.probability[cur_pos][3] / bg.0[3]).ln(),
                b'N' | b'n' => 0.0,
                _ => panic!("invalid nucleotide: {}", String::from_utf8(vec![seq[cur_pos + start]]).unwrap()),
            };
            cur_match += sc;
            let cur_best = cur_match + remain_best[cur_pos];

            if cur_best < thres {
                return None;
            } else if cur_pos >= n - 1 {
                return Some((start, cur_best));
            } else {
                cur_pos += 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DNAMotifScanner {
    pub motif: DNAMotif,
    cdf: ScoreCDF,
    background: BackgroundProb,
}

impl DNAMotifScanner {
    pub fn find<'a>(
        &'a self,
        seq: &'a [u8],
        pvalue: f64,
    ) -> MotifSites<'a>
    {
        let thres = self.cdf.prob_inverse(1.0 - pvalue);
        MotifSites {
            motif: &self.motif,
            sigma: self.motif.optimal_scores_suffix(&self.background),
            background: &self.background,
            seq: seq,
            cur_pos: 0,
            thres,
        }
    }
}

#[derive(Debug, Clone)]
struct ScoreCDF(Vec<(f64, f64)>);

impl ScoreCDF {
    /// Approximate the cdf of motif matching scores using dynamic programming.
    /// Algorithm:
    /// Scan the PWM from left to right. For each position $i$, compute a score
    /// density function $s_i$ such that $s_i(x)$ is the total number of sequences
    /// with score $x$.
    fn new(motif: &DNAMotif, bg: &BackgroundProb) -> Self {
        struct ScoreGetter {
            lowest: f64,
            step: f64,
        }
        impl ScoreGetter {
            fn get_sc(&self, i: usize) -> f64 { (i as f64 + 0.5) * self.step + self.lowest }
        }

        let precision = 1e-5;
        let init = (vec![1.0], ScoreGetter { lowest: 0.0, step: 0.0 });
        let (accum, getter) = motif.probability.iter().fold(init, |(accum, getter), probs| {
            let normalized_probs: Vec<f64> = probs.iter().zip(bg.0.iter())
                .map(|(p_fg, p_bg)| (p_fg / p_bg).ln()).collect();
            let (min_prob, max_prob) = normalized_probs.iter()
                .minmax().into_option().unwrap();
            let lowest = getter.get_sc(
                accum.iter().enumerate().skip_while(|(_, x)| **x == 0.0).next().unwrap().0
            ) + min_prob;
            let highest = getter.get_sc(
                accum.iter().enumerate().rev().skip_while(|(_, x)| **x == 0.0).next().unwrap().0
            ) + max_prob;
            if lowest < highest {
                let num_bins = ((highest - lowest) / precision).ceil().min(200000.0) as usize;
                let step = (highest - lowest) / num_bins as f64;
                let mut new_accum = vec![0.0; num_bins];
                accum.into_iter().enumerate().for_each(|(i, v)| if v != 0.0 {
                    let sc = getter.get_sc(i);
                    normalized_probs.iter().zip(bg.0.iter()).for_each(|(p_norm, p_bg)| {
                        let idx = (((sc + p_norm - lowest) / step).floor() as usize)
                            .min(num_bins - 1);
                        new_accum[idx] += v * p_bg;
                    });
                });
                (new_accum, ScoreGetter { lowest, step })
            } else {
                (accum, getter)
            }
        });

        // TODO: compress CDF
        let cdf = accum.into_iter().scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        }).enumerate().map(|(i, x)| (getter.get_sc(i), x))
            .chunk_by(|x| x.1).into_iter().flat_map(|(_, mut groups)| {
                let a = groups.next().unwrap();
                match groups.last() {
                    None => vec![a],
                    Some(b) => vec![a, b],
                }
            }).collect();
        ScoreCDF(cdf)
    }

    fn prob_inverse(&self, p: f64) -> f64 {
        if p > 1.0 || p < 0.0 {
            panic!("p must be in [0,1]");
        }
        let cdf = &self.0;
        let n = cdf.len();
        let i = cdf.binary_search_by(|x| x.1.partial_cmp(&p).unwrap())
            .unwrap_or_else(|x| x);
        if i >= n {
            panic!("impossible");
        } else if i == 0 {
            if p == cdf[0].1 {
                cdf[0].0
            } else {
                panic!("impossible");
            }
        } else {
            let (ix_a, p_a) = cdf[i-1];
            let (ix_b, p_b) = cdf[i];
            let w1 = (p_b - p) / (p_b - p_a);
            let w2 = (p - p_a) / (p_b - p_a);
            w1 * ix_a + w2 * ix_b
        }
    }
}

pub struct MotifSites<'a> {
    motif: &'a DNAMotif,
    sigma: Vec<f64>,
    background: &'a BackgroundProb,
    seq: &'a [u8],
    cur_pos: usize,
    thres: f64,
}

impl<'a> Iterator for MotifSites<'a> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur_pos + self.motif.size() >= self.seq.len() + 1 {
                return None;
            }
            let search_result = self.motif.look_ahead_search(
                self.background,
                &self.sigma,
                self.seq,
                self.cur_pos,
                self.thres,
            );
            self.cur_pos += 1;
            if search_result.is_some() {
                return search_result;
            }
        }
    }
}

pub fn parse_meme(content: &str) -> Vec<DNAMotif> {
    content.split("MOTIF").skip(1).map(|s| {
        let mut lines = s.lines();
        let id = lines.next().unwrap().trim().to_string();
        let mut iter = lines.skip_while(|x| !x.starts_with("letter-probability matrix"));
        let n: usize = iter.next().unwrap().split("w=").last().unwrap()
            .split("nsites=").next().unwrap().trim().parse().unwrap();
        let pwm = iter.take(n).map(|x| x.trim().split_ascii_whitespace()
            .map(|v| v.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        ).collect();
        DNAMotif { id, name: None, family: None, probability: pwm }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let bg = BackgroundProb::default();
        let motif1_str = "MOTIF 1_ASCCAGGCKGG
letter-probability matrix: alength= 4 w= 11 nsites= 14 E= 3.2e-035
0.768791 0.07577 0.120456 0.034983
0.057276 0.532522375 0.282802875 0.12739862500000002
0.048894125 0.796682375 0.09644475 0.05797825
0.044861750000000006 0.6375725 0.263928 0.05363775
0.61916725 0.104690625 0.17939824999999998 0.096743875
0.086404375 0.0745045 0.807213125 0.031878125
0.08837187499999999 0.07052475 0.828965875 0.012137499999999999
0.039856875 0.80558425 0.08105799999999999 0.07350025
0.079739875 0.110598625 0.40112975 0.40853125
0.03881925 0.12873025 0.781449 0.05100175
0.136913 0.15827100000000002 0.5818685 0.1229465";
        let scores = vec![7.009906220318511];

        let motif1: DNAMotif = motif1_str.parse().unwrap();
        let cdf = ScoreCDF::new(&motif1, &Default::default());
        //assert_eq!(cdf.prob_inverse(1.0 - 1e-4), scores[0]);
        //let seq = "ATATCGGCATACGATACGGACGGAT";
        let seq = "ATATCCCATCG";
        //motif1.look_ahead_search(&bg, &motif1.optimal_scores_suffix(&bg), seq.as_bytes(), 20, 0.0);
        let sites: Vec<_> = motif1.to_scanner(bg).find(seq.as_bytes(), 0.9).collect();
    }
}