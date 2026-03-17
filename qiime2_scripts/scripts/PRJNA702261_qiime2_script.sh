#!/bin/bash
qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' --input-path manifest_PRJNA702261 --output-path PRJNA702261.qza  --input-format PairedEndFastqManifestPhred33V2
qiime cutadapt trim-paired --i-demultiplexed-sequences PRJNA702261.qza --p-front-f CCTAYGGGRBGCASCAG --p-front-r GGACTACNNGGGTATCTAAT --o-trimmed-sequences PRJNA702261_trimmed
qiime quality-filter q-score --i-demux PRJNA702261_trimmed.qza --o-filtered-sequences PRJNA702261_trimmed_30_filtered.qza --o-filter-stats PRJNA702261_30 --p-min-quality 30
qiime vsearch merge-pairs --i-demultiplexed-seqs PRJNA702261_trimmed_30_filtered.qza --o-merged-sequences PRJNA702261_trimmed_30_filtered_merged --o-unmerged-sequences PRJNA702261_trimmed_30_filtered_unmerged
qiime deblur denoise-16S --i-demultiplexed-seqs PRJNA702261_trimmed_30_filtered_merged.qza --output-dir PRJNA702261_trimmed_253_deblur --p-trim-length 253
cd PRJNA702261_trimmed_253_deblur
qiime feature-classifier classify-sklearn --i-reads representative_sequences.qza --i-classifier silva-138-99-nb-classifier.qza --o-classification PRJNA702261_silva_reads
qiime taxa collapse --i-taxonomy PRJNA702261_silva_reads.qza --i-table table.qza --o-collapsed-table PRJNA702261_collapsed_table --p-level 7
qiime taxa barplot --i-taxonomy PRJNA702261_silva_reads.qza --i-table table.qza --o-visualization PRJNA702261_barplot
