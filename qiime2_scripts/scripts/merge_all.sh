#!/bin/bash


P1="PRJNA766410_trimmed_253_deblur"
P2="PRJNA702261_trimmed_253_deblur"
P3="PRJNA280490_trimmed_253_deblur"


qiime feature-table merge-seqs \
  --i-data $P1/representative_sequences.qza \
  --i-data $P2/representative_sequences.qza \
  --i-data $P3/representative_sequences.qza \
  --o-merged-data merged_rep_seqs.qza


qiime feature-table merge \
  --i-tables $P1/table.qza \
  --i-tables $P2/table.qza \
  --i-tables $P3/table.qza \
  --o-merged-table merged_table.qza



qiime feature-table merge \
  --i-tables $P1/PRJNA766410_collapsed_table.qza \
  --i-tables $P2/PRJNA702261_collapsed_table.qza \
  --i-tables $P3/PRJNA280490_collapsed_table.qza \
  --o-merged-table merged_collapsed_table.qza


qiime feature-table summarize \
  --i-table merged_table.qza \
  --o-visualization merged_table.qzv

qiime feature-table summarize \
  --i-table merged_collapsed_table.qza \
  --o-visualization merged_collapsed_table.qzv

