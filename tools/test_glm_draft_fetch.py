"""Tests for tools/glm_draft_fetch.py: the draft block is found in a GGUF's tensor directory, only its bytes are
read, and what is written is a GGUF of its own with the tensors as the source has them.  A fake GGUF in memory -
nothing is downloaded.

    python -m unittest tools.test_glm_draft_fetch
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import glm_draft_fetch as G  # noqa: E402


def text(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(tensors, align=32):
    """A GGUF with a string, a number and an array of strings as metadata, and `tensors` [(name, dims, bytes)]."""
    kv = text("general.architecture") + struct.pack("<I", 8) + text("glm5next")
    kv += text("general.alignment") + struct.pack("<II", 4, align)
    kv += text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3) + text("a") + text("bc") + text("def")
    kv += text("glm5next.swiglu_clamp_exp") + struct.pack("<IIQ2f", 9, 6, 2, 10.0, 7.5)     # one a block
    kv += text("glm5next.block_count") + struct.pack("<II", 4, 2)
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 5) + kv
    data = b""
    for name, dims, blob in tensors:
        out += text(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", 0, len(data))
        data += blob + b"\0" * (-len(blob) % align)
    return out + b"\0" * (-len(out) % align) + data


# The cut between the model's files goes through the draft block, as it does in the real ones: its attention is at
# the end of one file, the rest in the next.
GATE, NORM, PROJ, ATTN = bytes(range(200)) * 3, b"\x05" * 64, bytes(range(40, 104)), b"\x0b" * 64
PLAIN = gguf([("blk.0.attn_norm.weight", [16], b"\x01" * 64)])
EARLY = gguf([("blk.0.ffn_norm.weight", [16], b"\x03" * 64), ("blk.1.attn_norm.weight", [16], ATTN)])
SOURCE = gguf([("blk.1.ffn_gate_exps.weight", [10, 5, 3], GATE), ("blk.1.nextn.enorm.weight", [16], NORM),
               ("output.weight", [8, 2], b"\x02" * 64), ("blk.1.nextn.eh_proj.weight", [8, 2], PROJ)])


class Fetch(unittest.TestCase):
    def setUp(self):
        self.read = []                                    # the ranges asked for
        self.files = {"repo/1.gguf": PLAIN, "repo/2.gguf": EARLY, "repo/3.gguf": SOURCE}
        patch = mock.patch.multiple(G, REPO="repo/", SHARDS=["1.gguf", "2.gguf", "3.gguf"], PIECE=256, SHA256=None)
        patch.start()
        self.addCleanup(patch.stop)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = os.path.join(self.dir.name, "mtp", "draft.gguf")

    def get(self, url, start, end):
        self.read.append((url, start, end))
        return self.files[url][start:end + 1]

    def size(self, url):
        return len(self.files[url])

    def fetch(self):
        with mock.patch("builtins.print"):
            G.fetch(self.out, self.get, self.size)

    def block(self):
        with open(self.out, "rb") as f:
            data = f.read()
        tensors, start, self.meta = G.directory(data)
        return {n: (d, data[start + o:start + o + {"blk.1.ffn_gate_exps.weight": 600}.get(n, 64)])
                for n, d, _, o in tensors}

    def test_the_block_is_written_as_the_source_has_it(self):
        self.fetch()
        self.assertEqual(self.block(), {"blk.1.attn_norm.weight": ([16], ATTN),
                                        "blk.1.ffn_gate_exps.weight": ([10, 5, 3], GATE),
                                        "blk.1.nextn.enorm.weight": ([16], NORM),
                                        "blk.1.nextn.eh_proj.weight": ([8, 2], PROJ)})
        self.assertEqual(G.verify(self.out), 0)
        self.assertFalse(os.path.exists(self.out + ".part"))
        # the block's own SwiGLU limit goes with it, under the model's name for it
        self.assertEqual(self.meta["general.architecture"], "glm5next")
        self.assertEqual(self.meta["glm5next.swiglu_clamp_exp"], 7.5)
        self.assertNotIn("glm5next.block_count", self.meta)

    def test_only_the_directories_and_the_block_are_read(self):
        self.fetch()
        for name in ("repo/2.gguf", "repo/3.gguf"):
            tensors, start, _ = G.directory(self.files[name])
            other = [(start + o, start + o + 63) for n, _, _, o in tensors if not n.startswith("blk.1.")]
            for url, a, b in self.read:
                if url == name and a > 0:                 # a > 0: not a directory read
                    self.assertFalse(any(a <= hi and lo <= b for lo, hi in other), (a, b))
        self.assertEqual([r for r in self.read if r[0] == "repo/1.gguf" and r[1] > 0], [])

    def test_a_fetch_that_stopped_goes_on(self):
        self.fetch()
        with open(self.out, "rb") as f:
            whole = f.read()
        os.remove(self.out)
        with open(self.out + ".part", "wb") as f:
            f.write(whole[:len(whole) - 300])            # stopped inside the last tensor but one
        self.read.clear()
        self.fetch()
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), whole)
        gate = [r for r in self.read if r[0] == "repo/3.gguf" and r[1] > 0 and r[2] - r[1] == 255]
        self.assertLess(len(gate), 3)                     # the first pieces of the large tensor were kept

    def test_a_part_of_another_block_starts_over(self):
        os.makedirs(os.path.dirname(self.out))
        with open(self.out + ".part", "wb") as f:
            f.write(b"\x09" * 2000)
        self.fetch()
        self.assertEqual(self.block()["blk.1.nextn.eh_proj.weight"], ([8, 2], PROJ))

    def test_a_wrong_hash_is_refused(self):
        with mock.patch.multiple(G, SHA256="0" * 64, REVISION=G.PINNED_REVISION), self.assertRaises(SystemExit):
            self.fetch()
        self.assertFalse(os.path.exists(self.out))
        self.assertFalse(os.path.exists(self.out + ".part"))

    def test_no_block_anywhere(self):
        self.files["repo/3.gguf"] = PLAIN
        with self.assertRaises(SystemExit):
            self.fetch()


class Directory(unittest.TestCase):
    def test_a_directory_cut_short_asks_for_more(self):
        _, start, _ = G.directory(SOURCE)
        for n in (3, 30, 120, start - 40):
            with self.assertRaises(G.Short):
                G.directory(SOURCE[:n])

    def test_not_a_gguf(self):
        with self.assertRaises(ValueError):
            G.directory(b"PK\x03\x04" + b"\0" * 64)

    def test_verify(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.gguf")
            with mock.patch("builtins.print"), mock.patch.object(G, "SHA256", None):
                self.assertEqual(G.verify(path), G.BAD)   # missing
                with open(path, "wb") as f:
                    f.write(PLAIN)
                self.assertEqual(G.verify(path), G.BAD)   # a GGUF without the block
                with open(path, "wb") as f:
                    f.write(SOURCE)
                self.assertEqual(G.verify(path), 0)
            with mock.patch("builtins.print"), mock.patch.object(G, "REVISION", G.PINNED_REVISION):
                self.assertEqual(G.verify(path), G.BAD)   # not the pinned revision's bytes

    def test_the_block_is_pinned(self):
        self.assertRegex(G.SHA256, "^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
