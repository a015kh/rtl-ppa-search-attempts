"""Dispatch each benchmark adapter in its own configured process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description='EvolVE search: configured benchmark adapters with unchanged search logic.')
    parser.add_argument('workflow',choices=['icrtl','four-pipeline','verilogeval','cvdp','compare'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args,rest=parser.parse_known_args()
    if rest[:1]==['--']:rest=rest[1:]
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if '--help' not in rest and args.workflow != 'compare':
        output_flag='--output-root' if args.workflow=='verilogeval' else '--run-dir'
        if not any(x==output_flag or x.startswith(output_flag+'=') for x in rest):
            parser.error('Provide an explicit '+output_flag+' for search output.')
    if '--help' not in rest:
        benchmark_key = {'icrtl': 'benchmark_root', 'four-pipeline': 'benchmark_root',
                         'verilogeval': 'verilogeval_benchmark', 'cvdp': 'cvdp_benchmark'}.get(args.workflow)
        benchmark_flag = '--cvdp-benchmark' if args.workflow == 'cvdp' else '--benchmark-dir'
        overridden = args.workflow in {'verilogeval', 'cvdp'} and any(
            value == benchmark_flag or value.startswith(benchmark_flag + '=') for value in rest)
        if benchmark_key and not overridden and not config.get('paths', {}).get(benchmark_key):
            parser.error('Set an explicit paths.' + benchmark_key + ' in the configuration.')
    modules={'icrtl':'evolve','four-pipeline':'four_pipeline','verilogeval':'run_verilogeval_four_pipeline','cvdp':'cvdp','compare':'compare_proxy_accuracy'}
    if args.workflow in {'four-pipeline','verilogeval'} and '--help' not in rest:
        inference=config.get('inference',{})
        if not all(isinstance(inference.get(k),int) and inference[k]>0 for k in ['max_input_len','num_samples']):
            parser.error('Set positive inference.max_input_len and inference.num_samples explicitly.')
    command=[sys.executable,'-m','revolution.evolve.'+modules[args.workflow],*rest]
    env=os.environ.copy()
    env['REVOLUTION_EVOLVE_CONFIG']=str(args.config.resolve())
    src=Path(__file__).resolve().parents[2]
    env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','')
    env.pop('LD_LIBRARY_PATH',None)
    env.pop('PYTHONHOME',None)
    if args.dry_run:
        print(json.dumps({'command':command,'config':str(args.config.resolve())},indent=2))
        return 0
    return subprocess.call(command,env=env)

if __name__=='__main__':
    raise SystemExit(main())
