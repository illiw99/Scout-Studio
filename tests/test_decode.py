"""Automatic decode selection and continuity across a failed CUDA decode."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from scout import media


class DecodeTests(unittest.TestCase):
    def test_resize_decode_and_cpu_fallbacks_preserve_one_continuous_sequence(self):
        def failure(count):
            yield from range(count)
            raise RuntimeError('unsupported backend')
        with patch.object(media,'ffmpeg_has_cuda',return_value=True), \
             patch.object(media,'_iter_frames',side_effect=[failure(2),failure(4),iter(range(6))]) as decode, \
             warnings.catch_warnings(record=True) as notices:
            self.assertEqual(list(media.iter_frames('video',5,2,2,gpu_resize=True)),list(range(6)))
        self.assertEqual(len(notices),2)
        self.assertTrue(decode.call_args_list[0].kwargs['gpu_resize'])
        self.assertNotIn('gpu_resize',decode.call_args_list[1].kwargs)
        self.assertFalse(decode.call_args_list[2].kwargs.get('cuda',False))

    def test_cuda_resize_happens_before_download(self):
        import io
        from types import SimpleNamespace
        proc=SimpleNamespace(stdout=io.BytesIO(bytes(12)),stderr=io.BytesIO(),wait=lambda:0)
        with patch.object(media.subprocess,'Popen',return_value=proc) as start:
            self.assertEqual(len(list(media._iter_frames('video',5,2,2,cuda=True,gpu_resize=True))),1)
        cmd=start.call_args.args[0];filters=cmd[cmd.index('-vf')+1]
        self.assertIn('-hwaccel_output_format',cmd)
        self.assertLess(filters.index('scale_cuda'),filters.index('hwdownload'))

    def test_gpu_success(self):
        with patch.object(media, 'ffmpeg_has_cuda', return_value=True), \
             patch.object(media, '_iter_frames', return_value=iter([1, 2])) as decode:
            self.assertEqual(list(media.iter_frames('video', 5, 2, 2)), [1, 2])
        self.assertTrue(decode.call_args.kwargs['cuda'])
        self.assertEqual(decode.call_count, 1)

    def test_unavailable_or_disabled_uses_cpu(self):
        for supported, enabled in [(False, True), (True, False)]:
            with self.subTest(supported=supported, enabled=enabled), \
                 patch.object(media, 'ffmpeg_has_cuda', return_value=supported), \
                 patch.object(media, '_iter_frames', return_value=iter([1, 2])) as decode:
                self.assertEqual(list(media.iter_frames('video', 5, 2, 2,
                                                       hwaccel=enabled)), [1, 2])
                self.assertFalse(decode.call_args.kwargs.get('cuda', False))

    def test_cuda_failure_preserves_sequence(self):
        for before_failure in (0, 2):
            def gpu(*args, **kwargs):
                yield from range(before_failure)
                raise RuntimeError('driver unavailable')

            with self.subTest(before_failure=before_failure), \
                 patch.object(media, 'ffmpeg_has_cuda', return_value=True), \
                 patch.object(media, '_iter_frames', side_effect=[gpu(), iter(range(5))]), \
                 self.assertWarnsRegex(RuntimeWarning, 'using CPU'):
                self.assertEqual(list(media.iter_frames('video', 5, 2, 2)), list(range(5)))

    def test_real_cpu_fallback(self):
        # Force the unsupported CUDA path in the bundled build, then verify
        # the real reader still produces the requested RGB frames on CPU.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'video.mp4'
            media.run([media.ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                       'color=c=red:s=64x48:r=25:d=1', '-c:v', 'libx264', str(source)])
            with patch.object(media, 'ffmpeg_has_cuda', return_value=True), \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                frames = list(media.iter_frames(source, 5, 64, 48))
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].shape, (48, 64, 3))
            self.assertGreater(np.mean(frames[0][:, :, 0]), 200)
