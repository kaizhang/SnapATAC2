Input data format
=================

SnapATAC2 currently accepts BED-like tabular file as input. It will support BAM file or other formats in the future. The BED-like tabular file can be used to represent fragments or insertions.

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

Insertion records usually represent reads from the single-ended sequencing.
While fragment records are created by two transposition events, insertion records correspond to a single transposition event.

Each insertion record must contain six fields:

1. Reference genome chromosome.
2. Adjusted start position on chromosome.
3. Adjusted end position on chromosome. The end position is exclusive.
4. The cell barcode of this fragment.
5. The total number of reads associated with this insertion.
6. The strandness of the read.

During data import, either the start or the end position of an insertion record is converted to one insertion count depends on the strandness of the record.

Note: in both cases, the fifth column (duplication count) is not used during reads counting.
In other words, we count duplicated reads only once.
If you want to count the same record multiple times, you need to duplicate them in the input file.