#!/bin/bash
qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' --input-path manifest_PRJNA280490 --output-path PRJNA280490.qza  --input-format PairedEndFastqManifestPhred33V2
qiime cutadapt trim-paired --i-demultiplexed-sequences PRJNA280490.qza --p-front-f CCTAYGGGRBGCASCAG --p-front-r GGACTACNNGGGTATCTAAT --o-trimmed-sequences PRJNA280490_trimmed
qiime quality-filter q-score --i-demux PRJNA280490_trimmed.qza --o-filtered-sequences PRJNA280490_trimmed_30_filtered.qza --o-filter-stats PRJNA280490_30 --p-min-quality 30
qiime vsearch merge-pairs --i-demultiplexed-seqs PRJNA280490_trimmed_30_filtered.qza --o-merged-sequences PRJNA280490_trimmed_30_filtered_merged --o-unmerged-sequences PRJNA280490_trimmed_30_filtered_unmerged
qiime deblur denoise-16S --i-demultiplexed-seqs PRJNA280490_trimmed_30_filtered_merged.qza --output-dir PRJNA280490_trimmed_253_deblur --p-trim-length 253
cd PRJNA280490_trimmed_253_deblur
qiime feature-classifier classify-sklearn --i-reads representative_sequences.qza --i-classifier silva-138-99-nb-classifier.qza --o-classification PRJNA280490_silva_reads
qiime taxa collapse --i-taxonomy PRJNA280490_silva_reads.qza --i-table table.qza --o-collapsed-table PRJNA280490_collapsed_table --p-level 7
qiime taxa barplot --i-taxonomy PRJNA280490_silva_reads.qza --i-table table.qza --o-visualization PRJNA280490_barplot
