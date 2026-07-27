"""The one deliberate guardrail: no answer sentence ships unless entailed by its
cited source span.

For each claim, NLI-check against its citation; if not entailed, drop / regenerate
/ flag (config-driven; default: flag + regenerate once, then flag). Phase 2.6 —
not yet implemented.
"""
