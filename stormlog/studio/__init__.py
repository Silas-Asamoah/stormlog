"""Stormlog Studio: a local web workspace for inference artifacts (#227).

Studio is one loopback HTTP server that reads Stormlog artifacts under a
workspace root, computes every displayed value with Stormlog's own Python
code and serves a browser frontend that only formats and draws them.

The types its components share live in :mod:`stormlog.studio.contracts`.
"""
