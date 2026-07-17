# Prompt Engineering and RAG Quality

## Mitigating Hallucinations in RAG Systems

Retrieval-Augmented Generation (RAG) systems reduce hallucination through several
complementary strategies:

- **Retrieval grounding**: the model is instructed to base its answer only on the
  retrieved passages, rather than relying on parametric knowledge from pretraining.
- **Source citations**: requiring the model to cite which retrieved chunk supports
  each claim makes ungrounded statements easier to detect and discourages fabrication.
- **System prompt constraints**: explicit instructions such as "if the answer is not
  in the provided context, say you don't know" prevent the model from filling gaps
  with invented facts.
- **Temperature adjustments**: lowering the sampling temperature (often to 0) makes
  the model's outputs more deterministic and less likely to drift into speculative
  or fabricated content.

## Maximum Marginal Relevance (MMR)

Maximum marginal relevance (MMR) is a re-ranking technique used during document
retrieval to balance relevance and diversity among the chunks returned to the model.
A pure top-k similarity search can return several near-duplicate chunks that are all
highly similar to the query but redundant with each other.

MMR addresses this in a re-ranking phase: after an initial larger candidate set is
retrieved, MMR iteratively selects the next chunk that is both relevant to the query
and dissimilar to chunks already selected. This reduces redundancy in the final
context passed to the model, giving it broader coverage of the source material
instead of several near-identical passages.

## Chunking Strategy for Ingestion

When ingesting documents for retrieval, several parameters typically dictate the
chunking strategy:

- **Chunk size**: the target length of each chunk, often measured in tokens or
  characters, balancing enough context per chunk against retrieval precision.
- **Chunk overlap**: a number of tokens repeated between consecutive chunks so that
  information spanning a chunk boundary isn't lost entirely from either chunk.
- **Semantic boundaries**: splitting at natural boundaries such as paragraphs,
  sections, or sentences rather than at arbitrary character counts, so chunks remain
  coherent units of meaning.
- **Token limits**: the maximum number of tokens a chunk can contain, constrained by
  the embedding model's input limit and the context window available downstream.
