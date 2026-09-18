#!/usr/bin/env python3
PSP = 8 * 16 + 4 * 16 + 2 * 32
OURS = 10 * 16 + 2 * 48
assert PSP == 256
assert OURS == 256
print({"psp": PSP, "early10to2": OURS})
