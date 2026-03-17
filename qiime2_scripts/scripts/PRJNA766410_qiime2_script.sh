#!/bin/bash
qiime tools import   --type 'SampleData[SequencesWithQuality]'   --input-path manifest_PRJNA766410   --output-path PRJNA766410.qza   --input-format SingleEndFastqManifestPhred33V2
qiime quality-filter q-score --i-demux PRJNA766410.qza --o-filtered-sequences PRJNA766410_trimmed_30_filtered.qza --o-filter-stats PRJNA766410_30 --p-min-quality 30
qiime deblur denoise-16S --i-demultiplexed-seqs PRJNA766410_trimmed_30_filtered.qza --output-dir PRJNA766410_trimmed_253_deblur --p-trim-length 253
cd PRJNA766410_trimmed_253_deblur
qiime feature-classifier classify-sklearn --i-reads representative_sequences.qza --i-classifier silva-138-99-nb-classifier.qza --o-classification PRJNA766410_silva_reads
qiime taxa collapse --i-taxonomy PRJNA766410_silva_reads.qza --i-table table.qza --o-collapsed-table PRJNA766410_collapsed_table --p-level 7
qiime taxa barplot --i-taxonomy PRJNA766410_silva_reads.qza --i-table table.qza --o-visualization PRJNA766410_barplot
