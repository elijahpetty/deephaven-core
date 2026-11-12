---
title: What's the minimum amount of memory I should allocate to a Deephaven server?
---

We recommend allocating at least 4 GB of heap to the Deephaven server (`-Xmx4g`). The machine or container needs more total memory than that, because Groovy scripts and other off-heap memory are not part of the Java heap.

See [Configure and adjust heap size](../../how-to-guides/heap-size.md) to learn how to change heap settings for Docker and other deployments.

> [!NOTE]
> These FAQ pages contain answers to questions about Deephaven Community Core that our users have asked in our [Community Slack](/slack). If you have a question that is not in our documentation, [join our Community](/slack) and we'll be happy to help!
