"""Deterministic, section-aware TXT ingestion with exact line provenance."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = {'hr_policy': 'hr', 'compliance_manual': 'compliance',
         'tech_docs': 'product', 'onboarding_guide': 'onboarding'}

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def load_corpus(folder=None, max_chars=1800, overlap_lines=2):
    if max_chars < 200 or overlap_lines < 0:
        raise ValueError('Invalid chunk settings')
    chunks = []
    for path in sorted(Path(folder or ROOT / 'Sample Documents').glob('*.txt')):
        raw = path.read_text(encoding='utf-8')
        lines = raw.splitlines()
        header = '\n'.join(lines[:4])
        version = re.search(r'Version:?\s*([\d.]+)', header)
        date = re.search(r'(Effective Date|Last Updated):\s*([^|\n]+)', header)
        owner = re.search(r'(?:Owner|Reviewed by):\s*([^|\n]+)', header)
        if path.stem == 'onboarding_guide':
            owner_value = 'People Operations'
        else:
            owner_value = owner.group(1).strip() if owner else None
        base = {'document_id': path.stem, 'filename': path.name, 'title': lines[0],
                'document_type': TYPES.get(path.stem, 'other'),
                'version': version.group(1) if version else 'unknown',
                'content_hash': digest(raw), 'owner': owner_value,
                'ingested_at': datetime.now(timezone.utc).isoformat(),
                'source_path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name}
        if date:
            text = date.group(2).strip()
            for fmt in ('%B %d, %Y', '%B %Y'):
                try:
                    parsed = datetime.strptime(text, fmt)
                    base['effective_date' if date.group(1) == 'Effective Date' else 'updated_date'] = parsed.strftime('%Y-%m-%d' if fmt == '%B %d, %Y' else '%Y-%m')
                    break
                except ValueError:
                    pass
        pattern = r'^(\d+\.)\s+(.+)$' if path.stem == 'tech_docs' else r'^(\d+\.\d+)\s+(.+)$'
        starts = [(i, re.match(pattern, line))
                  for i, line in enumerate(lines)]
        starts = [(i, m) for i, m in starts if m]
        boundaries = [(0, None)] + starts + [(len(lines), None)]
        for n in range(len(boundaries) - 1):
            start, match = boundaries[n]
            stop = boundaries[n + 1][0]
            section = match.group(1).rstrip('.') if match else 'preamble'
            heading = match.group(2) if match else 'Document overview'
            cursor = start
            while cursor < stop:
                end = cursor
                size = 0
                while end < stop and (size + len(lines[end]) <= max_chars or end == cursor):
                    size += len(lines[end]) + 1
                    end += 1
                text = '\n'.join(lines[cursor:end]).strip()
                if text:
                    metadata = {k: v for k, v in base.items() if v is not None}
                    metadata.update(section_number=section, section_title=heading,
                                    line_start=cursor + 1, line_end=end)
                    cid = digest(f'{path.stem}:{base["content_hash"]}:{cursor}:{end}')[:32]
                    chunks.append({'id': cid, 'text': text, 'metadata': dict(metadata, chunk_id=cid)})
                cursor = max(cursor + 1, end - overlap_lines) if end < stop else stop
    if not chunks:
        raise ValueError('No TXT documents found')
    return chunks


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)
