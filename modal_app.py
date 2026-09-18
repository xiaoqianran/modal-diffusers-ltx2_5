"""Compatibility entrypoint.

The real Modal deployment lives in deploy/modal.py.
Use modal deploy deploy/modal.py for new workflows.
"""

from deploy.modal import *  # noqa: F401,F403
