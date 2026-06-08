#!/usr/bin/env python3
"""Generate a placeholder 1024x1024 app icon (no deps): dark bg + orange disc."""
import struct
import zlib
import sys

W = H = 1024
BG = (11, 11, 13)       # #0b0b0d
FG = (230, 85, 47)      # #e6552f
cx = cy = W / 2
r = 330
r2 = r * r

raw = bytearray()
for y in range(H):
    raw.append(0)  # filter type: none
    dy = y - cy
    for x in range(W):
        dx = x - cx
        if dx * dx + dy * dy <= r2:
            raw += bytes((*FG, 255))
        else:
            raw += bytes((*BG, 255))


def chunk(tag: bytes, data: bytes) -> bytes:
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")

out = sys.argv[1] if len(sys.argv) > 1 else "icon-1024.png"
with open(out, "wb") as f:
    f.write(png)
print(f"wrote {out} ({len(png)} bytes)")
