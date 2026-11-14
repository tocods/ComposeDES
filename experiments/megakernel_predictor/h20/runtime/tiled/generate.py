#!/usr/bin/env python3
"""Add weight tiles to the resident MoE experiment without changing its earlier plans."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tuning'))
import generate as base


def main(source, output):
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'tuned.cu'
        base.main(source, path)
        text = path.read_text()
    text = text.replace('#include <cooperative_groups.h>', '#include <cooperative_groups.h>\n#include <cuda/atomic>')
    text = text.replace('    int used;\n    int ids', '    int used;\n    unsigned int ready[slots];\n    int ids')
    text = text.replace('                r.groups[group].count = 0;',
                        '                r.groups[group].count = 0;\n                r.ready[group] = 0;')
    where = text.index('enum Stage')
    text = text[:where] + Path(__file__).with_name('weight-tiles.inc').read_text() + '\n' + text[where:]
    text = text.replace('        default:\n            GGML_ABORT("invalid resident MoE cut");',
                        '        case 7:\n        case 8:\n        case 9:\n'
                        '            launch_tiles<batch>(stream, blocks, s, mode, first, second);\n'
                        '            break;\n'
                        '        default:\n            GGML_ABORT("invalid resident MoE cut");')
    output.write_text(text)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
