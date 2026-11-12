---
title: How can I find which Core version added a particular feature or fix?
---

Every merged change in the [deephaven-core GitHub repository](https://github.com/deephaven/deephaven-core/commits/main/) has a commit on `main`. To find which release first included a change:

1. Search the commit list or pull-request list for the feature name, issue number, or PR number. Merge commits end with the PR number, such as `(#4934)`.
2. Click the commit title to open it.
3. Below the commit message, check the tags listed. The **earliest** release tag is the first Core version that includes the change. Any later tags shown are subsequent releases that also contain it.

<!-- TODO: update the screenshot to show a commit with multiple release tags so readers can see the earliest-tag pattern described above. -->

For example, on the following commit page, the earliest release tag shown is 42.7:

![A commit page with branches and tags highlighted](../../assets/reference/community-questions/commitmessage.png)

If the commit is on `main` and no release tag is listed, the change has not been released yet and will ship in the next release.

The deephaven-core repository holds only Core changes. Web UI changes are tagged in the [deephaven/web-client-ui](https://github.com/deephaven/web-client-ui) repository.
