"""A file for strata-poly --suffix-corpus: text files as token ids, a file a line.

    python tools/suffix_corpus.py TOKENIZER_DIR OUT.ids FILE [FILE ...]

TOKENIZER_DIR is a pack's tokenizer folder (for Qwen3.6 on this PC
C:\\tools\\PolyStrata-data\\packs\\qwen36-ud-q4_k_m\\tokenizer). Each FILE is encoded as plain text (special tokens
are not parsed) and written as one line of comma-separated ids. The engine then guesses from these files too, as
from the prompts and answers it has seen: `--suffix-drafts 256 --suffix-corpus OUT.ids`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tok = ST.Tokenizer.from_dir(Path(sys.argv[1]))
    total = 0
    with open(sys.argv[2], "w") as out:
        for name in sys.argv[3:]:
            ids = tok.encode(Path(name).read_text(encoding="utf-8", errors="replace"), parse_special=False)
            if ids:
                out.write(",".join(map(str, ids)) + "\n")
                total += len(ids)
    print(f"{sys.argv[2]}: {len(sys.argv) - 3} files, {total} tokens")


if __name__ == "__main__":
    main()
