# RBI retrieval

Semantic search over the RBI weekly reserve-money workbook, backed by Chroma.

## Build the index

    python -m app.retrieval.ingest

Reads `data/RBI_Data.xlsx`, writes `data/chroma/`. Re-running upserts on the
week-ending date, so it updates rather than duplicating.

## Query

    python -m app.retrieval.query

## Sheet structure

The workbook has a two-level header:

    row 4   merged group header ("Components", "Sources") -- NOT the column names
    row 5   the actual column names
    row 6   blank
    row 7+  weekly data, newest first
    col 0   empty throughout
    col 1   the date, with no header of its own

Taking row 4 as the header leaves 8 of 13 columns named `nan`, which produces
documents like `nan: 39542.5731`. Those embed and retrieve without error and are
meaningless -- a silent failure worth knowing about.

## What retrieval is and is not good for here

Good: "what was happening to currency in circulation around the 2016
demonetisation", "weeks where RBI's claims on banks went sharply negative" --
questions where you want to find relevant periods and read the figures.

Not good, and these fail quietly:

  - exact lookup -- may return a neighbouring week that embeds similarly
  - superlatives ("highest week") -- needs a scan and sort, not top-k similarity
  - arithmetic ("change between 2019 and 2020") -- retrieval does not compute

For those, add a repository with query/rank/compare functions over the same
data. Any figure that gets spoken aloud should come from that path, where it
traces to a row, rather than from a retrieved passage the model paraphrases.