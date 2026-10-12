from __future__ import annotations

import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vcu_surrogate.dataset import download_entry, read_range, split_ids, trace_group, zip_entry
from vcu_surrogate.rl_env import GridDataset, Normalizer, OfflineGOPEnv
from vcu_surrogate.traces import read_timestamp_mbps, throughput_to_bandwidth


def test_irregular_trace_integrates_and_retains_outage():
    samples = np.array([[0., 99.], [.5, 2.], [1.5, 4.], [2.5, 0.], [3.2, 2.]])
    np.testing.assert_allclose(throughput_to_bandwidth(samples), [3., 2., 1.])
    np.testing.assert_allclose(throughput_to_bandwidth(np.array([[0., 1.], [2., 0.]])), [0., 0.])


@pytest.mark.parametrize('samples', [[[0, 1], [0, 2]], [[0, 1], [1, -1]], [[0, 1], [1, float('nan')]], [[1, 1], [2, 1]]])
def test_invalid_trace_rejected(samples):
    with pytest.raises(RuntimeError):
        throughput_to_bandwidth(np.array(samples))


def test_timestamp_mbps_input_is_strict_and_preserves_outages(tmp_path):
    path = tmp_path / 'norway.log'
    path.write_text('0 9\n0.5 2\n1.5 0\n2.2 3\n')
    values = read_timestamp_mbps(path)
    np.testing.assert_array_equal(values[:, 1], [9, 2, 0, 3])
    path.write_text('0\n100\n200\n')
    with pytest.raises(RuntimeError, match='timestamp/Mbps'):
        read_timestamp_mbps(path)
    path.write_text('0 1\n1 bad\n')
    with pytest.raises(RuntimeError, match='timestamp/Mbps'):
        read_timestamp_mbps(path)
    path.unlink()
    with pytest.raises(RuntimeError, match='timestamp/Mbps'):
        read_timestamp_mbps(path)


def test_splits_are_order_independent_and_disjoint():
    ids = [f'video_{i}' for i in range(100)]
    a = split_ids(ids, 13)
    assert a == split_ids(ids[::-1], 13)
    assert [list(a.values()).count(s) for s in ('train', 'validation', 'test')] == [63, 7, 30]
    assert trace_group('trace_123_http---example') == trace_group('trace_123_http---other')
    assert trace_group('bus.trip_part3.log') == trace_group('bus.trip_part7.log')


def test_zip_mov_and_resumable_download(tmp_path):
    payload = b'video fixture bytes'
    name = b'bdd100k/videos/train/clip.mov'
    crc = zlib.crc32(payload)
    data = struct.pack('<IHHHHHIIIHH', 0x04034B50, 10, 0, 0, 0, 0, crc, len(payload), len(payload), len(name), 0) + name + payload
    reader = lambda start, size: data[start:start + size]
    entry = zip_entry(0, reader)
    assert entry['name'].endswith('.mov')
    path = tmp_path / 'clip.mov'
    path.with_suffix('.mov.part').write_bytes(payload[:5])
    with patch('vcu_surrogate.dataset.run') as decoder:
        download_entry(entry, path, reader)
        assert decoder.call_count == 1
    assert path.read_bytes() == payload
    download_entry(entry, path, reader)
    path.write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='CRC'):
        download_entry(entry, path, reader)


def test_range_rejects_full_file_response():
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    with patch('urllib.request.urlopen', return_value=Response()):
        with pytest.raises(RuntimeError, match='HTTP Range'):
            read_range('https://example.test/video', 0, 10)


def test_source_sampling_is_balanced_and_train_only():
    dataset = GridDataset.__new__(GridDataset)
    dataset.trace_by_split = {'train': {'f': np.ones(20), **{f'n{i}': np.ones(20) for i in range(8)}},
                              'test': {'held': np.ones(20) * 999}}
    dataset.trace_source = {'f': 'fcc', **{f'n{i}': 'norway' for i in range(8)}, 'held': 'fcc'}
    rng = np.random.default_rng(13)
    choices = [dataset.sample_trace('train', 10, rng) for _ in range(10000)]
    assert .48 < choices.count('f') / len(choices) < .52
    assert 'held' not in choices
    dataset.block_seconds = 10
    _, lineage = dataset.synthetic(25, rng)
    assert sum(row['length'] for row in lineage) == 25
    assert all(row['trace_id'] != 'held' for row in lineage)


def test_environment_rejects_wrong_video_split():
    class Dataset:
        trace_by_split = {'train': {}}
        video_by_split = {'train': ['train'], 'test': ['test']}
        action_count = 32
    env = OfflineGOPEnv(Dataset(), 'train', Normalizer(np.zeros(6), np.ones(6), 10), 13)
    with pytest.raises(RuntimeError, match='video not in train'):
        env.reset(options={'video_id': 'test'})


def test_corrupt_zip_payload_is_not_published(tmp_path):
    payload = b'corrupt video'
    entry = {'data_offset': 0, 'size_bytes': len(payload), 'crc32': zlib.crc32(b'original video')}
    destination = tmp_path / 'bad.mov'
    with pytest.raises(RuntimeError, match='CRC'):
        download_entry(entry, destination, lambda start, size: payload[start:start + size])
    assert not destination.exists()
    assert not destination.with_suffix('.mov.part').exists()


def test_budget_rejected_before_network_request(tmp_path):
    from vcu_surrogate.dataset import download_bdd
    with patch('vcu_surrogate.dataset.archive_range') as reader:
        with pytest.raises(RuntimeError, match='byte budget'):
            download_bdd({'dataset': {'target_bytes': 4_000_000_000, 'max_bytes': 4_000_000_000}})
        reader.assert_not_called()


def test_trace_source_checksum_is_verified(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from vcu_surrogate.traces import assert_trace_sources
    from vcu_surrogate.common import sha256
    raw, processed = tmp_path / 'raw', tmp_path / 'processed'
    source = raw / 'network_traces' / 'trace'
    source.parent.mkdir(parents=True)
    processed.mkdir()
    source.write_text('0 1\n1 2\n')
    record = {'trace_id': 'fcc_trace', 'source_file': 'trace', 'raw_sha256': sha256(source)}
    config = {'paths': {'raw_data': str(raw), 'processed_data': str(processed)},
              'traces': {'url': 'fixture', 'commit': 'revision'}}
    (processed / 'trace_sources.json').write_text(json.dumps({'url': 'fixture', 'commit': 'revision', 'files': [record]}))
    monkeypatch.setattr('vcu_surrogate.traces.run', lambda *args, **kwargs: SimpleNamespace(stdout='revision\n'))
    assert assert_trace_sources(config) == [record]
    source.write_text('0 1\n1 3\n')
    with pytest.raises(RuntimeError, match='checksum'):
        assert_trace_sources(config)


def test_evaluation_uses_only_requested_video_split():
    dataset = GridDataset.__new__(GridDataset)
    dataset.semantic_by_video = {'train': np.zeros(6), 'val': np.zeros(6), 'test': np.zeros(6)}
    dataset.video_by_split = {'train': ['train'], 'validation': ['val'], 'test': ['test']}
    dataset.trace_by_split = {'validation': {'v': np.ones(12)}, 'test': {'t': np.ones(6)}}
    assert {s['video_id'] for s in dataset.evaluation_specs('validation')} == {'val'}
    assert dataset.evaluation_specs('test') == [{'video_id': 'test', 'trace_id': 't', 'trace_start': 0, 'synthetic': False}]
    dataset.trace_by_split['test']['t'] = np.ones(5)
    with pytest.raises(RuntimeError, match='long enough'):
        dataset.evaluation_specs('test')


def test_streamed_parquet_preserves_parts_and_is_atomic(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd
    from vcu_surrogate.parquet_stream import ParquetStream
    path = tmp_path / 'data.parquet'
    frames = [pd.DataFrame({'key': [str(i)], 'value': [float(i)]}) for i in range(3)]
    with ParquetStream(path) as writer:
        for frame in frames:
            writer.append(frame)
            assert not path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(path), pd.concat(frames, ignore_index=True))
    expected = path.read_bytes()
    with pytest.raises(ValueError):
        with ParquetStream(path) as writer:
            writer.append(frames[0])
            raise ValueError('interrupted')
    assert path.read_bytes() == expected
    assert not path.with_suffix('.tmp.parquet').exists()


def test_grid_dataset_enforces_manifest_identity_and_train_normalizer(tmp_path, monkeypatch):
    pytest.importorskip('pyarrow')
    import json
    import pandas as pd
    from types import SimpleNamespace
    from vcu_surrogate.common import sha256
    from vcu_surrogate.dataset import dataset_hash
    from vcu_surrogate.rl_env import STATE_COLUMNS
    from vcu_surrogate.traces import parse_throughput_trace
    raw, processed, artifacts = (tmp_path / name for name in ('raw', 'processed', 'artifacts'))
    for path in (raw / 'network_traces', processed, artifacts / 'metadata', artifacts / 'metrics', artifacts / 'encodes'):
        path.mkdir(parents=True)
    config = {'paths': {'raw_data': str(raw), 'processed_data': str(processed), 'artifacts': str(artifacts)},
              'traces': {'url': 'fixture', 'commit': 'revision'},
              'video': {'gop_frames': 30, 'fps': 30},
              'encoder': {'bitrates_mbps': [.5, 1, 1.5, 2, 3, 4, 6, 8], 'roi_delta_qp': [0, -3, -6, -9]},
              'reward': {}, 'rl': {'synthetic_block_seconds': 10, 'state_clip': 10}}
    videos = [{'video_id': split, 'split': split, 'source_sha256': split} for split in ('train', 'validation', 'test')]
    (processed / 'video_manifest.json').write_text(json.dumps({'videos': videos}))
    records, trace_rows = [], []
    for split_index, split in enumerate(('train', 'validation', 'test')):
        for source, base_bw in (('fcc', 2), ('norway', 4)):
            trace_id = f'{source}_{split}'
            path = raw / 'network_traces' / trace_id
            path.write_text(f'0 0\n12 {base_bw + split_index * 100}\n')
            record = dict(trace_id=trace_id, source_file=trace_id, source=source, split=split,
                          raw_sha256=sha256(path), group_id=trace_id)
            records.append(record)
            trace_rows.extend({**record, 'gop_idx': k, 'timestamp_s': float(k), 'bandwidth_mbps': float(value)}
                              for k, value in enumerate(parse_throughput_trace(path)))
    (processed / 'trace_sources.json').write_text(json.dumps({'url': 'fixture', 'commit': 'revision', 'files': records}))
    pd.DataFrame(records).to_parquet(processed / 'trace_catalog.parquet', index=False)
    pd.DataFrame(trace_rows).to_parquet(processed / 'bandwidth_traces.parquet', index=False)
    actions = [dict(action_id=i * 4 + j, target_bitrate_mbps=b, delta_qp_roi=q)
               for i, b in enumerate(config['encoder']['bitrates_mbps']) for j, q in enumerate(config['encoder']['roi_delta_qp'])]
    pd.DataFrame(actions).to_parquet(artifacts / 'encodes/action_space.parquet', index=False)
    grid, semantics = [], []
    for video in videos:
        offset = 0 if video['split'] == 'train' else 100
        for k in range(6):
            semantics.append(dict(video_id=video['video_id'], gop_idx=k, semantic_score=(k + 1) / 10))
            for action in actions:
                grid.append(dict(**action, video_id=video['video_id'], gop_idx=k, bitrate_index=action['action_id'] // 4,
                                 roi_index=action['action_id'] % 4, actual_bitrate_mbps=1 + action['action_id'] / 100 + offset,
                                 vmaf=80., psnr=35., ssim=.9, roi_ssim=.8, roi_psnr=34., roi_frame_count=1,
                                 frame_count=30, packet_count=30, key_packets=1))
    pd.DataFrame(grid).to_parquet(artifacts / 'metrics/gop_action_grid.parquet', index=False)
    pd.DataFrame(semantics).to_parquet(artifacts / 'metadata/gop_semantics.parquet', index=False)
    schema = {'schema_version': 2, 'gop_frames': 30, 'fps': 30, 'state_columns': list(STATE_COLUMNS),
              'action_count': 32, 'dataset_sha256': dataset_hash(config), 'reward_config': config['reward']}
    (processed / 'transition_schema.json').write_text(json.dumps(schema))
    monkeypatch.setattr('vcu_surrogate.rl_env.assert_downloads', lambda config: None)
    monkeypatch.setattr('vcu_surrogate.traces.run', lambda *args, **kwargs: SimpleNamespace(stdout='revision\n'))
    dataset = GridDataset(config)
    np.testing.assert_allclose(dataset.normalizer().mean[:4], [3.] * 4)
    assert dataset.normalizer().mean[4] == pytest.approx(1.155)
    assert {spec['video_id'] for spec in dataset.evaluation_specs('test')} == {'test'}
    changed_reward = {**config, 'reward': {**config['reward'], 'bandwidth_violation': .3}}
    with pytest.raises(RuntimeError, match='reward config mismatch'):
        GridDataset(changed_reward)
    traces_path = processed / 'bandwidth_traces.parquet'
    trace_table = pd.read_parquet(traces_path)
    trace_table.drop(columns=['timestamp_s']).to_parquet(traces_path, index=False)
    with pytest.raises(RuntimeError, match='no fallback'):
        GridDataset(config)
    bad_timestamp = trace_table.copy()
    bad_timestamp.loc[0, 'timestamp_s'] = .5
    bad_timestamp.to_parquet(traces_path, index=False)
    with pytest.raises(RuntimeError, match='timestamps'):
        GridDataset(config)
    trace_table.to_parquet(traces_path, index=False)
    schema['dataset_sha256'] = 'wrong-dataset'
    (processed / 'transition_schema.json').write_text(json.dumps(schema))
    with pytest.raises(RuntimeError, match='manifest hash mismatch'):
        GridDataset(config)
