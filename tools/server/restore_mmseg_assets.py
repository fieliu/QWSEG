"""Restore the tokenizer vocabulary from the matching upstream MMSeg wheel."""
import argparse
import gzip
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-url', default='https://pypi.tuna.tsinghua.edu.cn/simple')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    member = 'mmseg/utils/bpe_simple_vocab_16e6.txt.gz'
    target = root / 'seg/mmsegmentation-main-rgbt' / member
    if target.exists():
        gzip.decompress(target.read_bytes())
        print(f'Vocabulary already present: {target}')
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                        '--index-url', args.index_url, '--dest', tmp,
                        'mmsegmentation==1.2.2'], check=True)
        with zipfile.ZipFile(next(Path(tmp).glob('*.whl'))) as wheel:
            content = wheel.read(member)
        gzip.decompress(content)
        target.write_bytes(content)
    print(f'Restored vocabulary: {target}')


if __name__ == '__main__':
    main()
