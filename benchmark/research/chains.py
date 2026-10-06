"""Can ONE embedding of a compound request retrieve ALL the steps it needs?

    python -m benchmark.research.chains

This is the decisive question for multi-step routing architecture.

  If YES -- a single embedding surfaces every constituent operation in its
  top-k -- then "plan-then-route" works: one embedding call, the orchestrator
  returns a DAG, and the router's per-step latency cost (517 ms embed +
  1346 ms argument fill) is paid ONCE for the whole chain.

  If NO -- the embedding collapses onto whichever operation dominates the
  phrasing -- then plan-then-route is impossible with a single vector. The
  request must be decomposed into steps FIRST (an LLM call), and each step
  routed separately, so router overhead scales linearly with chain length and
  the context advantage has to pay for k embeddings instead of one.

Also measures:
  * error compounding -- if each step routes at accuracy p, a k-step chain
    succeeds at p^k only if step errors are independent. They may not be.
  * chain templates -- whether indexing a composite ("extract then summarize")
    as its own capability retrieves better than either part alone.

Chain requests below are authored to be genuinely compound: each names two or
three operations that must happen in order, phrased the way a user would.
"""

from __future__ import annotations

import numpy as np

from benchmark.research.common import Corpus, normalize
from orchestrator import config, embeddings

# (request, ordered operations it requires)
CHAINS: list[tuple[str, list[str]]] = [
    ("Read config.yaml and tell me what it says in plain English",
     ["FILE_READ", "TEXT_SUMMARIZE"]),
    ("Pull the text out of this PDF and give me a short summary",
     ["PDF_TEXT_EXTRACT", "DOCUMENT_SUMMARIZE"]),
    ("Find emails from the vendor and summarize what they want",
     ["EMAIL_SEARCH", "TEXT_SUMMARIZE"]),
    ("Search my calendar for next week and email me the list",
     ["CALENDAR_SEARCH", "EMAIL_SEND"]),
    ("Look up papers on tool routing and compare their methods",
     ["PAPER_SEARCH", "PAPER_COMPARE"]),
    ("Check the git status and commit whatever is staged",
     ["GIT_STATUS", "GIT_COMMIT"]),
    ("Read the log file and count how many errors it mentions",
     ["FILE_READ", "TEXT_KEYWORDS"]),
    ("Fetch that web page and pull out all the links",
     ["WEB_FETCH", "WEB_LINK_EXTRACT"]),
    ("Query the orders table and chart the totals by region",
     ["SQL_QUERY", "DATABASE_AGGREGATE"]),
    ("Translate this document into Spanish and save it to disk",
     ["TEXT_TRANSLATE", "FILE_WRITE"]),
    ("Find the issue about the crash and post a comment on it",
     ["ISSUE_SEARCH", "ISSUE_COMMENT"]),
    ("Geocode this address and tell me what timezone it is in",
     ["GEOCODE", "TIMEZONE_LOOKUP"]),
    ("Load the dataset and show me which columns have missing values",
     ["DATASET_INSPECT", "DATASET_MISSING_VALUES"]),
    ("Read the JSON file and check it against this schema",
     ["FILE_READ", "JSON_VALIDATE"]),
    ("Search the channel history and reply in that thread",
     ["MESSAGE_SEARCH", "MESSAGE_THREAD_REPLY"]),
    ("Resize this image and convert it to webp",
     ["IMAGE_RESIZE", "IMAGE_FORMAT_CONVERT"]),
    ("List the branches then create a new one off main",
     ["GIT_BRANCH_LIST", "GIT_CREATE_BRANCH"]),
    ("Get the sentiment of these reviews and email the summary to the team",
     ["TEXT_SENTIMENT", "EMAIL_SEND"]),
    ("Copy the report to the archive folder and then delete the original",
     ["FILE_COPY", "FILE_DELETE"]),
    ("Extract the product prices from this page and compute the average",
     ["WEB_STRUCTURED_EXTRACT", "STATISTICS_DESCRIBE"]),
    ("Read the CSV, filter to last quarter, and group by product",
     ["DATASET_INSPECT", "DATASET_FILTER", "DATASET_GROUP"]),
    ("Search for the paper, get its citations, and summarize the findings",
     ["PAPER_SEARCH", "PAPER_CITATIONS", "PAPER_SUMMARIZE"]),
    ("Find the email, read it, and draft a reply",
     ["EMAIL_SEARCH", "EMAIL_READ", "EMAIL_DRAFT"]),
    ("Fetch the page, extract the text, and translate it to Hindi",
     ["WEB_FETCH", "WEB_TEXT_EXTRACT", "TEXT_TRANSLATE"]),
]


def embed_cached(texts: list[str], tag: str) -> np.ndarray:
    path = config.CACHE_DIR / f"research_{tag}.npy"
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    print(f"  embedding {len(texts)} chain requests -> {path.name}")
    vectors = normalize(embeddings.embed_texts(texts, task_type="RETRIEVAL_QUERY"))
    np.save(path, vectors)
    return vectors


def main() -> int:
    corpus = Corpus()
    known = [(text, steps) for text, steps in CHAINS
             if all(s in corpus.op_index for s in steps)]
    if len(known) != len(CHAINS):
        print(f"skipping {len(CHAINS)-len(known)} chains referencing unknown operations")

    C = embed_cached([t for t, _ in known], "chain_queries")
    scores = C @ corpus.V.T
    order = np.argsort(-scores, axis=1)

    print(f"\n{len(known)} compound requests | {corpus.n_ops} operations\n")

    print("Can ONE embedding of the whole request find EACH required step?")
    print(f"{'step position':<16}{'rank@1':>9}{'top-3':>9}{'top-5':>9}{'top-10':>9}{'median rank':>13}")
    print("-" * 65)

    by_position: dict[int, list[int]] = {}
    for qi, (_, steps) in enumerate(known):
        for pos, op in enumerate(steps):
            rank = int(np.where(order[qi] == corpus.op_index[op])[0][0]) + 1
            by_position.setdefault(pos, []).append(rank)

    for pos in sorted(by_position):
        ranks = np.array(by_position[pos])
        print(f"step {pos+1:<11}{np.mean(ranks==1)*100:>8.1f}%{np.mean(ranks<=3)*100:>8.1f}%"
              f"{np.mean(ranks<=5)*100:>8.1f}%{np.mean(ranks<=10)*100:>8.1f}%"
              f"{int(np.median(ranks)):>13}")

    all_ranks = np.concatenate([np.array(v) for v in by_position.values()])
    print(f"{'ALL steps':<16}{np.mean(all_ranks==1)*100:>8.1f}%{np.mean(all_ranks<=3)*100:>8.1f}%"
          f"{np.mean(all_ranks<=5)*100:>8.1f}%{np.mean(all_ranks<=10)*100:>8.1f}%"
          f"{int(np.median(all_ranks)):>13}")

    # The decisive number for plan-then-route: are ALL steps of a chain present
    # in one top-k list simultaneously?
    print("\nAre ALL of a chain's steps inside a single top-k? "
          "(this is what plan-then-route needs)")
    for k in (3, 5, 10, 20):
        complete = 0
        for qi, (_, steps) in enumerate(known):
            topk = set(order[qi][:k].tolist())
            if all(corpus.op_index[s] in topk for s in steps):
                complete += 1
        print(f"  top-{k:<3} complete chains: {complete}/{len(known)}  "
              f"({complete/len(known)*100:.0f}%)")

    # Which step wins the single embedding?
    print("\nWhich step does the single embedding actually land on?")
    winner = {0: 0, 1: 0, 2: 0, -1: 0}
    for qi, (_, steps) in enumerate(known):
        top = order[qi][0]
        hit = -1
        for pos, op in enumerate(steps):
            if corpus.op_index[op] == top:
                hit = pos
                break
        winner[hit] = winner.get(hit, 0) + 1
    for pos, count in sorted(winner.items()):
        label = "an operation not in the chain at all" if pos == -1 else f"step {pos+1}"
        print(f"  {label:<38} {count:>3}/{len(known)}")

    print("\nError compounding, assuming independent steps at measured per-step accuracy:")
    for p, name in ((0.9564, "current flat"), (0.9762, "tuned blend")):
        row = "  ".join(f"k={k}: {p**k*100:5.1f}%" for k in (1, 2, 3, 5))
        print(f"  {name:<14} {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
