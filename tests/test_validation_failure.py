import os, sys, asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nl_sql_generator.autonomous_job import AutonomousJob

class DummyClient:
    def remaining_budget(self):
        return 1.0

    async def arun_jobs(self, *_, **__):
        return [{"role": "assistant", "content": '{"sql": "SELECT 1"}'}]

class FailingValidator:
    def check(self, sql):
        return False, "bad"

class DummyWriter:
    def fetch(self, *a, **k):
        return []
    def append_jsonl(self, *a, **k):
        pass

def test_run_task_handles_validation_failure():
    job = AutonomousJob({}, client=DummyClient(), validator=FailingValidator(), writer=DummyWriter(), critic=None)
    result = asyncio.run(job.run_task({"phase": "demo", "question": "q", "metadata": {}}))
    assert result.sql == "FAIL"
