Input data format
=================

SnapATAC2 accepts BAM or BED-like tabular file as input. The BED-like tabular file can be used to represent fragments (paired-end sequencing) or insertions (single-end sequencing). BAM files can be converted to BED-like
files using `pp.make_fragment_file`.

## Fragment interval format

Fragments are created by two separate transposition events, which create the two ends of the observed fragment.
Each unique fragment may generate multiple duplicate reads.
These duplicate reads are collapsed into a single fragment record.
**A fragment record must contain exactly five fields**:

1. Reference genome chromosome of fragment.
2. Adjusted start position of fragment on chromosome.
3. Adjusted end position of fragment on chromosome. The end position is exclusive, so represents the position immediately following the fragment interval.
4. The cell barcode of this fragment.
5. The total number of read pairs associated with this fragment. This includes the read pair marked unique and all duplicate read pairs.

During data import, a fragment record is converted to two insertions corresponding
to the start and end position of the fragment interval.

## Insertion format

Insertion records are used to represent single-end reads in experiments that sequence only one end of the fragments, e.g., Paired-Tag experiments.
While fragment records are created by two transposition events, insertion records correspond to a single transposition event.

Each insertion record must contain six fields:

1. Reference genome chromosome.
2. Adjusted start position on chromosome.
3. Adjusted end position on chromosome. The end position is exclusive.
4. The cell barcode of this fragment.
5. The total number of reads associated with this insertion.
6. The strandness of the read.

During data import, the 5' end of an insertion record is converted to one insertion count.

Note: in both cases, the fifth column (duplication count) is not used during reads counting.
In other words, we count duplicated reads only once.
If you want to count the same record multiple times, you need to duplicate them in the input file.