# Embeddings

Embeddings define how the agent observes the environment. It must convert
the graph into a fixed-sized vector representation. There are two types
of embeddings possible: atomic and composite.

Atomic embeddings operate on the graph itself. Composite embeddings operate
on other embeddings.

## `ChainEmbedding`

An atomic embedding that embeds the sum of the longest chains possible from
all non-directed donors.

## `DdEmbedding`

An atomic embedding that embeds the number of directed donors.

## `NddEmbedding`

An atomic embedding that embeds the number of non-directed donors.

## `NopEmbedding`

An atomic embedding that is empty.

## `OrderEmbedding`

An atomic embedding that embeds the order of the graph.

## `UnionEmbedding`

A composite embedding that unions other embeddings.

## `Walk2VecEmbedding`

An atomic embedding that performs a modified Walk2Vec random walk
procedure.