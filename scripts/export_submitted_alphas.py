#!/usr/bin/env python3
"""Compatibility wrapper for the submitted alpha recorder tool."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_agent.submitted_alpha_tool import main


if __name__ == "__main__":
    main()
