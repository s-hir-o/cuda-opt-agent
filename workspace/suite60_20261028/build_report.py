"""Build the final Word report from saved measurements; never synthesizes results."""
from pathlib import Path
import collections
import hashlib
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'report_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

COLORS = {'improved':'#16856b','unchanged':'#8893a1','inconclusive':'#e4a42f',
          'regressed':'#ca4b4b','failed':'#832a43'}

def add_table(doc, headers, rows, font=8):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Light Shading Accent 1'
    for cell, text in zip(table.rows[0].cells, headers): cell.text = str(text)
    repeat = OxmlElement('w:tblHeader')
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for cell, text in zip(table.add_row().cells, row): cell.text = str(text)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs: run.font.size = Pt(font)
    return table

def chart(doc, path, caption):
    doc.add_picture(str(path), width=Inches(6.5))
    doc.add_paragraph(caption, 'Caption')

def main():
    results = json.loads((ROOT/'results.json').read_text())
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert len(results) == 60 and len({r['case'] for r in results}) == 60, 'Need all 60 results'
    assert {r['case'] for r in results} == {r['case'] for r in manifest}
    counts = collections.Counter(r['status'] for r in results)
    measured = [r for r in results if 'timings' in r and 'speedup' in r]
    improved = [r for r in measured if r['status'] == 'improved']
    checks = sum(len(r.get('cpu_checks', [])) for r in results)
    failed = [r for r in results if r['status'] == 'failed']
    saved_audit = json.loads((ROOT/'audit'/'saved_results_audit.json').read_text())
    figures = ROOT/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold'})
    fig, ax = plt.subplots(figsize=(10,3.4))
    statuses = list(COLORS)
    bars = ax.bar(statuses, [counts[s] for s in statuses], color=[COLORS[s] for s in statuses])
    ax.bar_label(bars); ax.set_ylim(0,max(counts.values())+5)
    ax.set_ylabel('Cases'); ax.set_title('Outcomes from 60 real CLI optimization runs')
    fig.tight_layout(); fig.savefig(figures/'outcomes.png',dpi=180); plt.close(fig)
    for part in range(2):
        subset = results[part*30:(part+1)*30]
        fig, ax = plt.subplots(figsize=(10,10))
        for i,r in enumerate(subset):
            if 'speedup' in r:
                low,high = r['speedup_ci95']; mid=r['speedup']
                ax.plot([low,high],[i,i],color=COLORS[r['status']],lw=2)
                ax.scatter([mid],[i],color=COLORS[r['status']],s=28,zorder=3)
        ax.axvline(1,color='#374151',ls='--',lw=1)
        ax.set_yticks(range(len(subset)),[r['case'] for r in subset],fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('Speedup = baseline mean / output mean (higher is faster)')
        ax.set_title(f'Independent CUDA-event timings: cases {part*30+1}–{(part+1)*30}')
        ax.grid(axis='x',alpha=.2); fig.tight_layout()
        fig.savefig(figures/f'speedups_{part+1}.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    for status,color in COLORS.items():
        subset=[r for r in measured if r['status']==status]
        if subset: ax.scatter([r['timings']['baseline']['mean_ms'] for r in subset],
                              [r['timings']['optimized']['mean_ms'] for r in subset],
                              color=color,label=status,alpha=.8)
    if measured:
        vals=[r['timings'][v]['mean_ms'] for r in measured for v in ('baseline','optimized')]
        ax.plot([min(vals),max(vals)],[min(vals),max(vals)],'--',color='#374151')
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Baseline latency (ms, log scale)'); ax.set_ylabel('Output latency (ms, log scale)')
    ax.set_title('Points below the diagonal have lower measured latency'); ax.legend()
    fig.tight_layout(); fig.savefig(figures/'latency.png',dpi=180); plt.close(fig)

    doc=Document()
    sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65)
    sec.left_margin=sec.right_margin=Inches(.75)
    normal=doc.styles['Normal']; normal.font.name='Calibri'; normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6)
    for style in ('Title','Heading 1','Heading 2'):
        doc.styles[style].font.color.rgb=RGBColor.from_string('173B57')
    sec.header.paragraphs[0].text='CUDA OPTIMIZATION  |  60-CASE EVALUATION'
    foot=sec.footer.paragraphs[0]; foot.text='Measured locally • 28 October 2026  |  '
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); foot._p.append(fld)
    doc.add_heading('CUDA optimization and correctness audit',0)
    doc.add_paragraph('60 real CLI cases • 12 kernel families • NVIDIA GeForce RTX 2080 Ti', 'Subtitle')
    doc.add_paragraph('Prepared 29 October 2026 from the 28 October evaluation evidence. Synthetic representative CUDA workloads written for this evaluation; not NVIDIA source code or an assessment of NVIDIA engineering standards.')
    doc.add_heading('Measured outcome',1)
    doc.add_paragraph(f"Attempted {len(results)} CLI cases: {counts['improved']} confirmed improvements, "
                      f"{counts['inconclusive']} inconclusive, {counts['unchanged']} unchanged, "
                      f"{counts['regressed']} confirmed regressions and {counts['failed']} failures. "
                      f"The independent harness recorded {checks} successful CPU-reference comparisons.")
    doc.add_paragraph(f"All-60 correctness was NOT established. Only {len(measured)} of 60 cases have the complete six-check baseline/output CPU validation and independent timing record. "
                      f"The remaining {len(failed)} cases failed the source guard before independent validation; no independent speedup is claimed for them. "
                      f"The saved-buffer audit independently passed {saved_audit['cases_passed']}/{saved_audit['cases_checked']} case records and "
                      f"{saved_audit['buffer_checks_passed']}/{saved_audit['buffer_checks']} available buffer comparisons.")
    if improved:
        best=max(improved,key=lambda r:r['speedup'])
        doc.add_paragraph(f"Largest confirmed measured gain: {best['case']}, {best['speedup']:.3f}× "
                          f"({100*(1-1/best['speedup']):.1f}% lower kernel latency). "
                          f"Its 95% bootstrap interval is {best['speedup_ci95'][0]:.3f}–{best['speedup_ci95'][1]:.3f}×.")
    else: doc.add_paragraph('No changed output met the predeclared improvement threshold in these measurements.')
    if measured:
        gm=math.exp(sum(math.log(r['speedup']) for r in measured)/len(measured))
        doc.add_paragraph(f"Descriptive geometric mean of measured speedup ratios across {len(measured)} cases: {gm:.3f}×. "
                          'This includes unchanged outputs and timing noise; it is not a general expected speedup or an independent-sample statistical claim.')
    doc.add_paragraph('The correctness audit found gaps in the CLI verification layer despite passing existing tests. '
                      'Performance conclusions here rely on the additional independent harness checks; see the audit findings and their limits below.')
    chart(doc,figures/'outcomes.png','Figure 1. Unchanged source is never counted as an optimization, regardless of timing noise.')
    doc.add_heading('Scope and execution environment',1)
    doc.add_paragraph('Two specialist agents handled benchmark construction/execution and independent verification auditing. '
                      'The primary agent assembled the evidence and report. Existing repository changes were retained.')
    add_table(doc,['Item','Value'],[
        ['GPU','NVIDIA GeForce RTX 2080 Ti, 11 GB; Windows WDDM; sm_75'],
        ['Compiler','CUDA nvcc 13.2, V13.2.51; Windows MSVC toolchain'],
        ['Driver','616.92 (nvidia-smi at start of evaluation)'],
        ['Repository HEAD','3b61030fa0140f4290357f1a7959fedf5d6a72ab, working tree modified before evaluation'],
        ['Optimizer','Actual cuda_opt_agent.cli.app optimize invocation per case; max_iter=18'],
        ['Generation','Built-in transformation search; no authenticated LLM generation'],
        ['Profiling','GPU hardware-counter access unavailable; no profiler-derived efficiency claims'],
    ])
    doc.add_heading('Workload design',1)
    doc.add_paragraph('Each baseline is a straightforward functional implementation with ordinary untuned choices: direct global-memory access, serial work within a thread, or missing tiling/vectorization. '
                      'No artificial sleeps or deliberately serial whole-vector kernel is used. Some simple pointwise kernels may already be close to practical efficiency. '
                      'The 60 cases are 12 algorithms at five sizes, not 60 independent algorithms.')
    fams=list(dict.fromkeys(m['family'] for m in manifest))
    add_table(doc,['Family','Sizes (N)','Baseline design'],[
        [f,', '.join(str(m['size']) for m in manifest if m['family']==f),next(m['description'] for m in manifest if m['family']==f)] for f in fams])
    doc.add_heading('Measurement and verification protocol',1)
    doc.add_paragraph('For each case the real CLI receives a complete host-runnable CUDA translation unit and writes its selected source. '
                      'The independent harness recompiles baseline and selected output. Both versions are checked against a NumPy float64 CPU oracle '
                      'at (N, seed 0), (N+3, seed 1), and (N−1, seed 7), including sizes not divisible by the usual block/tile size. '
                      'Checks require the expected output length, finite float32 values and elementwise agreement with rtol=1e−4 and atol=1e−5.')
    doc.add_paragraph('Inputs use deterministic uint32 hashing to generate signed dyadic values. The suite checks the host contract and kernel signature, '
                      'records SHA-256 source hashes and rejects nonfinite or nonpositive timing samples. The independent auditor re-reads saved output buffers '
                      'with separately expressed CPU oracles and checks source integrity and timing arithmetic. These checks reduce trust in candidate-controlled output.')
    doc.add_paragraph('An explicitly reviewed exception permits pragma-unroll annotations in the host harness; other host or signature changes are rejected. '
                      'The exact CLI-produced source is compiled and timed. Host launch-loop unrolling can affect launch pacing within a CUDA-event interval, '
                      'so such gains describe the complete CLI-produced implementation, not strictly device-kernel instruction changes. '
                      'The ten saved failed cases remain failed even where their host edits appear to be pragma-only: no recovery execution or independent validation was completed for those cases.')
    doc.add_paragraph('Timing uses 12 samples per version, alternating baseline/output order; each sample performs 5 warmup launches and times 30 launches with CUDA events. '
                      'Reported speedup is mean baseline milliseconds divided by mean selected-output milliseconds. '
                      'Intervals are percentile 95% bootstrap intervals from 10,000 independent resamples of the 12 timings per version (seed 42). '
                      'A changed source is classified improved only when the lower bound exceeds 1.01; regressed when the upper bound is below 0.99; otherwise inconclusive.')
    doc.add_paragraph('These are kernel timings, not application end-to-end latency. Windows display activity, clock changes and launch scheduling can affect results. '
                      'Intervals are per case, not adjusted for 60 comparisons, and do not establish generalization to other GPUs. '
                      'A CLI-selected variant can fail to improve under independent measurement; every such outcome is retained.')
    doc.add_page_break()
    doc.add_heading('Optimization graphs',1)
    chart(doc,figures/'speedups_1.png','Figure 2. Measured speedup with 95% bootstrap intervals, first 30 cases. Green: improved; grey: unchanged; amber: inconclusive; red: regressed.')
    doc.add_page_break()
    chart(doc,figures/'speedups_2.png','Figure 3. Remaining 30 cases, using the same classification and timing method.')
    chart(doc,figures/'latency.png','Figure 4. Baseline versus selected-output kernel latency. A point below the diagonal alone does not establish a confirmed improvement.')
    doc.add_heading('All 60 case results',1)
    rows=[]
    for r in results:
        t=r.get('timings')
        rows.append([r['case'],r['status'],f"{t['baseline']['mean_ms']:.6f}" if t else '—',
                     f"{t['optimized']['mean_ms']:.6f}" if t else '—',f"{r['speedup']:.3f}" if t else '—',
                     f"{r['speedup_ci95'][0]:.3f}–{r['speedup_ci95'][1]:.3f}" if t else '—'])
    add_table(doc,['Case','Outcome','Base ms','Out ms','×','95% interval'],rows,7)
    doc.add_heading('Failed cases and missing evidence',1)
    doc.add_paragraph('Each case below exited the CLI with code 0 but was rejected by the suite host/signature guard. '
                      'A source export or successful CLI exit does not establish correctness or speedup. '
                      'The saved-buffer auditor identifies host unroll-pragma edits for these ten records, but this narrow exception does not supply the missing output buffers or timings. '
                      'These are validation failures, not demonstrated numerical mismatches. Their CLI-reported speedups are excluded from the performance charts and aggregates.')
    add_table(doc,['Failed case','Selected variant','Recorded failure'],[
        [r['case'],r.get('selected_variant','—'),r.get('error','See case log')] for r in failed],8)
    doc.add_heading('Results by kernel family',1)
    family_rows=[]
    for family in fams:
        subset=[r for r in results if r['family']==family]
        c=collections.Counter(r['status'] for r in subset)
        family_rows.append([family,c['improved'],c['unchanged'],c['inconclusive'],c['regressed'],c['failed']])
    add_table(doc,['Family','Improved','Unchanged','Uncertain','Regressed','Failed'],family_rows,8)
    doc.add_heading('Selected transformations',1)
    changed=[r for r in results if r.get('source_changed')]
    add_table(doc,['Case','CLI-selected variant','Independent outcome'],[
        [r['case'],r.get('selected_variant','See case log'),r['status']] for r in changed],8)
    if not changed: doc.add_paragraph('The CLI retained the baseline for every case.')
    doc.add_page_break()
    doc.add_heading('Independent correctness audit',1)
    audit_path=ROOT/'audit'/'AUDIT.md'
    assert audit_path.exists(), 'Final audit required'
    # Preserve the auditor’s detailed evidence, including explicit scope limitations.
    for line in audit_path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        if line.startswith('#'):
            doc.add_heading(line.lstrip('# ').strip(),2)
        else:
            doc.add_paragraph(line.replace('**','').replace('`',''))
    doc.add_heading('Reproduction and evidence',1)
    doc.add_paragraph('From the repository root, run .venv\\Scripts\\python.exe -m workspace.suite60_20261028.run_suite. '
                      'The runner resumes saved completed cases; use a separate output directory for a fresh repeat. '
                      'Build this report with python workspace/suite60_20261028/build_report.py.')
    doc.add_paragraph('The suite directory contains manifest.json, results.json, results.csv, run_suite.py, audit/ and figures/. '
                      'Each case preserves input.cu, output.cu, command.json, config.yaml, optimize.log and result.json; compiled binaries and correctness buffers exist where execution reached those stages. '
                      'Audit files preserve existing test output and adversarial probe evidence. '
                      'Source and result SHA-256 hashes are recorded in evidence_sha256.json.')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and 'report_dependencies' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.json','.csv','.cu','.py','.md','.log') and p.name not in ('evidence_sha256.json','report_validation.json')]
    (ROOT/'evidence_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    out=ROOT/'CUDA_60_Case_Optimization_Report.docx'; doc.save(out)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml'): ET.fromstring(z.read(name))
        media=[name for name in z.namelist() if name.startswith('word/media/')]
        assert len(media)>=4
    reopened=Document(out)
    text='\n'.join(p.text for p in reopened.paragraphs)+'\n'+'\n'.join(c.text for t in reopened.tables for row in t.rows for c in row.cells)
    assert all(r['case'] in text for r in results)
    validation={'report':str(out),'bytes':out.stat().st_size,'cases':len(results),'counts':dict(counts),
                'cpu_checks':checks,'embedded_graphs':len(media),'zip_crc_valid':True,'xml_well_formed':True,
                'python_docx_reopen_passed':True,'all_60_case_names_present':True,
                'all_60_cpu_correctness_established':False if failed else True,
                'word_render_succeeded':False,
                'word_render_limitation':'Word COM activation failed with 0x80070520: specified logon session does not exist. Visual pagination not verified.',
                'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    (ROOT/'report_validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))

if __name__=='__main__': main()
