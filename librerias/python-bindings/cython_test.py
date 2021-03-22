#!/usr/bin/env python
import pylibfromc

# Sample data for our call
x, y = 6, 2.3

answer = pylibfromc.pymult(x, y)
print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
