# Context: RAS 6-page conversion of the cross-bank fraud detection paper

This file exists so a Claude session opening this repo has the same
background as the one that did this conversion. Read this before touching
`paper/manuscript/root.tex`.

## What this is

The IEEE-RAS conference version of "A Federated Expert-Fusion Framework for
Cross-Bank Fraud Detection" (Patil, Li, Ordoñez Chala, Nguyen, Ling — UTS).
Submission deadline: 28 Aug 2026, via ras.papercept.net. Hard limit: 6 pages.

- `paper/manuscript/root.tex` — the RAS-template source, currently compiles
  to exactly 6 pages
- `paper/manuscript/ieeeconf.cls` — the official RAS class file, downloaded
  from https://ras.papercept.net/conferences/support/tex.php (ieeeconf.zip)
- `paper/figures/fig1.png` — architecture diagram (three-phase pipeline)
- `paper/figures/fig2.png` — ULB AUPRC vs Dirichlet alpha
- `paper/figures/fig3.png` — family-level AUPRC bar chart at alpha=0.5
- `paper/manuscript/root.pdf` — last verified compile (6 pages, no errors,
  no undefined refs, no overfull boxes)

The original, non-RAS 8-page draft (IEEEtran class, not this class) is a
separate file elsewhere and is NOT in this repo. This repo holds only the
RAS-converted version.

## Why it needed converting

The original draft used `\documentclass[conference,a4paper]{IEEEtran}`.
Steve's email said to use the RAS/PaperCept template instead, which is a
different class (`ieeeconf`, not `IEEEtran`) with a different macro set.
`ieeeconf.cls` does NOT define `\IEEEauthorblockN`/`\IEEEauthorblockA` or the
`IEEEkeywords` environment — those had to be rewritten, not just the
`\documentclass` line. Reference: the official sample in
`ieeeconf.zip` (Albert Author / Bernard D. Researcher) shows the correct
`\author{...$^{1}$...\thanks{$^{1}$...}}` pattern.

## Page budget: how 8 pages became 6

Original compiled to 8 pages on IEEEtran. Two things closed the gap:

1. **Fig. 1 moved from a double-column `figure*` to a single-column
   `figure`** at `\columnwidth`. This was the single biggest win — a
   full-width figure at any width (tested 0.46, 0.70, 0.85, 1.0 \textwidth)
   forces a 7th page. Single-column packs far more efficiently even though
   the image itself is now smaller.
2. Prose trimmed roughly 25-30% across every section (not concentrated in
   one place — the author explicitly asked for cuts to be spread evenly).
   One float was removed outright: the "Best Deployable vs Centralised"
   table, whose three rows were already restated in adjacent prose.
3. Bibliography: 12 multi-author entries shortened to `et al.` form.
   Reference count is unchanged (still 35 entries, r1-r36 minus r26).
4. Float packing loosened (`topnumber`, `bottomnumber`, `totalnumber`,
   `topfraction`, etc.) beyond LaTeX defaults to reduce forced page breaks.

Verified empirically, not by eyeballing: every page-count claim in this repo
was checked by an actual `pdflatex` compile, not estimated.

## Author block

Three sub-affiliations, not one, per the author's clarification mid-project:

- Patil, Li, Ordoñez Chala — Master's students, FEIT (general)
- D. T. Nguyen — PhD student, School of Biomedical Engineering
- S. H. Ling — Associate Professor, School of Electrical, Mechanical and
  Biomedical Engineering (corresponding author, steve.ling@uts.edu.au)

All five have their student/staff email listed, matching the pattern in the
official RAS sample (which lists an email per author, not just one
corresponding address). This was a deliberate change back from an earlier
draft in this conversion that used one shared email only — Tommy and Steve
asked for all emails restored.

## Open items not yet decided

- **Fig. 1 size**: currently single-column (`\columnwidth`) to hit 6 pages.
  The author has asked about making it full-width/bigger since it reads
  small in print. Tested: full-width costs 1 page at any scale. The
  cheapest way to buy that page back without touching prose is dropping one
  table — dropping Table V (`tab:privacy`, the noise/sparsification AUC
  grid) frees a full spare page (verified 5 pages), vs dropping Table IV
  (`tab:survival`) which also works but is the RQ3 headline artifact.
  Table V is the better candidate because Section III-D's own text says to
  read that grid "for deltas rather than absolute levels" — the absolute
  numbers in the table are explicitly flagged as non-load-bearing by the
  paper itself. **Not yet applied to root.tex — pending author decision.**
- **Name spelling**: source has `Nicol\'as Ordo\~nez Chala`. The more common
  Spanish spelling of the surname is "Ordóñez" (accent on the o as well).
  His email is unaccented either way, so it doesn't disambiguate. Worth
  confirming directly with him before submission.
- **Funding footnote**: the official RAS sample includes
  `\thanks{*This work was not supported by any organization}`. This paper
  currently has no funding statement. Not added because it costs ~2 lines
  the page budget doesn't currently have spare; add if the venue requires
  a funding disclosure statement.
- **Corresponding author wording**: currently reads
  "Correspondence: steve.ling@uts.edu.au" — could read "Corresponding
  author:" instead if that's the more accurate role. One-word change,
  confirm with Steve.

## Verification checklist for any future edit to root.tex

Re-run all of these before treating a change as done — do not eyeball a
rendered page and guess the count:

1. Compile twice with `pdflatex -interaction=nonstopmode root.tex` (LaTeX
   needs two passes to resolve cross-references).
2. Check the log for zero of: `^!` (errors), `undefined` (refs/citations),
   `Overfull \vbox`/`\hbox`.
3. Confirm page count from the log line `Output written on root.pdf (N
   pages, ...)`.
4. If page count changes, re-check last-page column balance (RAS asks for
   manually balanced columns on the final page before camera-ready).
