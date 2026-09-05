# AGENTS.md — MoEArc

Working notes for anyone (human or agent) making changes here.

## Searching: stay inside the repo

🔴 **Never run `find /`.** Everything this project needs is in the repo or in two known
directories. A filesystem-wide search walks every mounted volume — including archive storage
that may be slow, large, or both — to find a file that was already within reach.

This is not hypothetical: a `find / -name ggml_dequant_dump` cost several minutes of head seeks
across multi-terabyte spinning storage, hunting a tool that lives in `tools/` in this repo.

Where things are:

| what | where |
| --- | --- |
| the code, tools, benchmarks, traces | this repo |
| models | the directory given in the task, never searched for |
| llama.cpp (oracle + baselines) | a sibling checkout, path given in the task |

Search `git ls-files`, or `find` **rooted at the repo or an explicitly given path**. If you
cannot find something, ask rather than widening the search — a missing path is a question, not
a reason to crawl the filesystem.
