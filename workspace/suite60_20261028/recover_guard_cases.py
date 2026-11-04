"""Finish exact-CLI-output checks rejected by the initial overly strict host guard.
Only full-line numeric unroll annotations may differ outside the work kernel.
Preserves CLI logs and initial guard rejection evidence; never reruns the optimizer.
"""
import asyncio
import json
from workspace.suite60_20261028.run_suite import ROOT,run_case,save_summary,ensure_windows_msvc_environment

async def main():
    assert ensure_windows_msvc_environment()
    results=json.loads((ROOT/'results.json').read_text())
    for i,result in enumerate(results):
        if result.get('error') == 'Candidate changed host harness or kernel signature':
            print(f"RECOVER {i+1}/60 {result['case']}",flush=True)
            results[i]=await run_case(result['family'],result['size'],reuse_cli=True)
            save_summary(results)
            print(f"DONE {result['case']}: {results[i]['status']} {results[i].get('error','')}",flush=True)
    print('RECOVERY COMPLETE',flush=True)

if __name__=='__main__': asyncio.run(main())
