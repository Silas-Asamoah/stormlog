"""Building blocks for Stormlog's exporters, with no knowledge of what they export.

The inference exporters (``stormlog.infer.export``) map records onto these
pieces: bounded queues, capped envelopes, a metric registry with fixed
budgets, a bounded ``/metrics`` server, a textfile writer, a line file sink,
a destination resolver and a socket watchdog. Each one bounds what it holds
and counts what it drops.
"""
