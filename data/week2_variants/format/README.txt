CV Matcher - Week 2 Formatting Robustness Variants

Selected pairs: PAIR_13, PAIR_10, PAIR_29, PAIR_09, PAIR_27, PAIR_01, PAIR_15, PAIR_11, PAIR_20, PAIR_25, PAIR_17, PAIR_14

For each selected source CV, four controlled variants are provided:
- *_single_column.pdf: canonical single-column flow
- *_two_column.pdf: two-column newspaper-style flow
- *_table.pdf: table-based section/content layout
- *_section_order.pdf: same single-column style, with detected top-level CV sections reversed

Controls:
- The original CV PDF remains the CTRL condition and is not duplicated in this folder.
- No substantive facts or lexical content were added or removed in the four generated variants.
- F1/F2/F3 preserve section order. F4 intentionally changes section order while keeping section-internal content unchanged.
- Typography is held constant across generated variants as far as possible so the manipulation focuses on layout/structure.

QA:
- 48/48 generated PDFs passed lexical content-multiset comparison against the source CV text extracted with PyMuPDF.
- qa.json contains the automated content checks.
- manifest.json maps experiment IDs to files and sources.
- Representative single-column, two-column, table, and section-order PDFs were rendered and visually checked for clipping/overlap.
