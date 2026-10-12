"""Bounded-memory Parquet output, published only after every part succeeds."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetStream:
    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_suffix('.tmp.parquet')
        self.writer = None
        self.rows = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        return self

    def append(self, frame):
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temporary, table.schema)
        self.writer.write_table(table)
        self.rows += len(frame)

    def __exit__(self, kind, value, traceback):
        if self.writer is not None:
            self.writer.close()
        if kind is None and self.writer is not None:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)
