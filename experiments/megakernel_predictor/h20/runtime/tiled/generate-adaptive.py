#!/usr/bin/env python3
"""Add an experimental work queue to the resident tiled MoE."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main(source, output):
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'tiled.cu'
        generator = Path(__file__).with_name("generate.py")
        subprocess.run([sys.executable, generator, source, path], check=True)
        text = path.read_text()
    text = text.replace('    unsigned int ready[slots];',
                        '    unsigned int ready[slots], next_down[slots], unclaimed;')
    text = text.replace('                r.ready[group] = 0;',
                        '                r.ready[group] = 0;\n                r.next_down[group] = 0;')
    text = text.replace("dst->name[17] - '0'", 'std::atoi(dst->name + 17)')
    where = text.index('enum Stage')
    text = text[:where] + Path(__file__).with_name('adaptive.inc').read_text() + '\n' + text[where:]
    text = text.replace('        default:\n            GGML_ABORT("invalid resident MoE cut");',
                        '        case 10:\n        case 11:\n'
                        '            launch_queued<batch>(stream, blocks, s, mode, first);\n'
                        '            break;\n'
                        '        default:\n            GGML_ABORT("invalid resident MoE cut");')
    output.write_text(text)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
