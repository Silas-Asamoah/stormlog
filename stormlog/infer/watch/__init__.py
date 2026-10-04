"""``stormlog infer watch``: bounded incident capture beside a vLLM server.

The watcher keeps a bounded record of a server's recent past and, when a
condition has been bad for long enough, seals what it has into an incident
bundle. This package holds its parts; see ``docs/incident_capture.md``.
"""
