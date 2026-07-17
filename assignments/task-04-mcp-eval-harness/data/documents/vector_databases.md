# Vector Databases and Semantic Search

## HNSW: Hierarchical Navigable Small World

Hierarchical Navigable Small World (HNSW) is the indexing algorithm used by most
modern vector databases to perform approximate nearest neighbor search over
high-dimensional embeddings. Its primary role is to make similarity search over
millions of embeddings fast, avoiding the need to compare a query vector against
every stored vector one by one.

### Why hierarchical structure matters

HNSW organizes vectors into a multi-layer graph. The top layers contain only a
sparse subset of points and are used for fast global search — the algorithm can
quickly jump across large distances in the vector space using long-range
connections, similar to how skip lists speed up search in sorted linked lists.

As the search descends through lower layers, the graph becomes denser, and the
algorithm refines its position with local convergence, moving to progressively
closer neighbors until it reaches the bottom layer, which contains every point.

This layered design lets HNSW achieve approximate nearest neighbor search in
sub-linear time relative to the number of stored embeddings, rather than the
linear time required by brute-force comparison.

## Dense vs. Sparse Embeddings

Semantic search systems generally rely on two families of representations:

- **Dense embeddings** are high-dimensional vectors (typically 384–1536 dimensions)
  produced by neural network encoders. They capture semantic meaning — two pieces
  of text with similar meaning will have vectors that are close together, even if
  they share no exact words.

- **Sparse embeddings** (such as those used in BM25) rely on exact keyword matching.
  Each dimension typically corresponds to a specific term in the vocabulary, and
  relevance is scored based on term frequency and overlap between the query and
  the document, rather than learned semantic similarity.

Many production retrieval systems combine both: dense embeddings for semantic
recall, and sparse/BM25-style scoring for precise keyword matching, then merge
the two result sets (hybrid search).
