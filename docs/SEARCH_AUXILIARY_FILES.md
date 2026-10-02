# Auxiliary benchmark files in population search

The recovered population-search implementation had an empty
`EoHEngine._copy_misc_files` method. Its intended copying loop was unreachable
after `_get_bypass_responses` returned. The existing unit test exposed this
problem. Some flat RTLLM-style testbenches read problem-specific memory or golden
data files from the candidate's working directory; saving only the generated RTL
could therefore leave those inputs missing during simulation.

The release restores copying when `_save_result_to_file` prepares each candidate
directory. Population search and the inherited single-shot workflow both use
this path. Functional simulation, synthesis checks, acceptance rules, scoring,
and model losses are unchanged. This preparation fix does not establish which
historical runs were affected and does not repair or relabel their saved results.

Only regular files directly in the configured benchmark directory are eligible.
Their names must begin with the exact problem name followed by `_` or `.`.
For example, `problem_test_data.dat`, `problem_reference.txt`, `problem_ref.dat`,
`problem_test.txt`, and `problem.hex`
are eligible; files for `problem2` are excluded when the selected name is
`problem`. Directories and symlinks are skipped. RTL source/include files,
the concrete historical suffixes `_prompt.txt`, `_ppa.txt`, `_ifc.txt`,
`_makefile`, `_ref.sv`, `_test.sv`, `_compiled.vvp`, `_simulation.log`, and
`_ref.syn.v`, together with generated thought, feedback, and diff records, are
excluded. A `ref` or `test` stem alone does not exclude non-RTL golden inputs.

The output directory must exist inside the configured result root, outside the
benchmark inputs, and use no symlink directories. Destination creation is
exclusive: existing regular files are preserved, while existing links or other
non-regular destinations produce an error. Use a new result directory after
changing benchmark inputs. On POSIX, directory descriptors anchor file access to
the checked source and output directories.

This adapter handles flat auxiliary files only. It does not recursively stage
ICRTL's `00_TB` hierarchy or discover files referenced through arbitrary paths
inside RTL. ICRTL evaluators retain their separate benchmark working-directory
rules. Benchmark payloads remain external to this public repository.
