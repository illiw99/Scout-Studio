"""Versioned worker jobs preserve literals, results and cancellation failures."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scout import worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root/'job.config.json'
        self.config.write_text('{}')

    def test_success_has_ordered_events_literal_options_and_bound_result(self):
        job = worker.new_job('cut', self.config, {'name_pattern':'Play <N> = final', 'start_number':3})
        path = self.root/'job.json'
        worker.write_job(path, job)
        events, calls = [], []
        def dispatch(arguments):
            calls.append(arguments)
            print('50.0% complete', flush=True)
            return 0
        self.assertEqual(worker.run(path, dispatch, events.append), 0)
        self.assertEqual(calls[0], ['--config',str(self.config),'cut',
                                    '--name-pattern=Play <N> = final','--start-number=3'])
        self.assertEqual([event['sequence'] for event in events], list(range(len(events))))
        self.assertEqual(events[0]['event'], 'started')
        self.assertEqual(events[-1]['event'], 'finished')
        self.assertIn(.5, [event['fraction'] for event in events if event['event']=='progress'])
        result = json.loads(worker.result_path(path,job).read_text())
        self.assertTrue(worker.validate_result(result,job)['ok'])
        with self.assertRaisesRegex(ValueError, 'already has a result'):
            worker.run(path, dispatch, events.append)

    def test_invalid_schema_commands_and_options_cannot_dispatch(self):
        valid = worker.new_job('run', self.config)
        for changes in ({'schema':2}, {'command':'train'}, {'config':'relative.json'},
                        {'options':{'force':True}}, {'job_id':'../escape'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                worker.validate_job(dict(valid,**changes))

    def test_exception_and_interrupt_record_failure(self):
        for error, code in ((RuntimeError('broken decoder'),1),(KeyboardInterrupt(),130)):
            with self.subTest(error=type(error).__name__):
                job=worker.new_job('run',self.config)
                path=self.root/f'{job["job_id"]}.json'
                worker.write_job(path,job)
                events=[]
                def dispatch(arguments):
                    raise error
                self.assertEqual(worker.run(path,dispatch,events.append),code)
                result=json.loads(worker.result_path(path,job).read_text())
                self.assertFalse(worker.validate_result(result,job)['ok'])
                self.assertEqual(result['exit_code'],code)

    def test_mismatched_or_inconsistent_results_fail(self):
        job=worker.new_job('run',self.config)
        result=dict(schema=1, job_id=job['job_id'], command='run', ok=True, exit_code=0)
        for changes in ({'schema':2},{'schema':True},{'schema':1.0},{'job_id':'0'*32},{'command':'cut'},
                        {'ok':False},{'exit_code':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                worker.validate_result(dict(result,**changes),job)

    def test_duplicate_running_descriptor_is_refused(self):
        job=worker.new_job('run',self.config)
        path=self.root/'job.json'
        worker.write_job(path,job)
        (self.root/f'{job["job_id"]}.running').write_text('fixture')
        with self.assertRaises(FileExistsError):
            worker.run(path,lambda args:0,lambda event:None)
