---
title: How do I configure Deephaven Community Core to use another port?
---

How you change the port depends on how you run Deephaven.

**Docker** — Change the host side of the port mapping and leave the container port at `10000`. For example:

```bash
docker run -p 8080:10000 ghcr.io/deephaven/server:latest
```

With Docker Compose, change the `ports` entry (for example, `"8080:10000"`) or set the `DEEPHAVEN_PORT` environment variable. See [Change the port](../../getting-started/docker-install.md#change-the-port) for details.

**Other deployments** — Set the `http.port` [configuration property](../../how-to-guides/configuration/configuration-properties.md), for example with `START_OPTS=-Dhttp.port=8080` or in your configuration file.

> [!NOTE]
> These FAQ pages contain answers to questions about Deephaven Community Core that our users have asked in our [Community Slack](/slack). If you have a question that is not in our documentation, [join our Community](/slack) and we'll be happy to help!
