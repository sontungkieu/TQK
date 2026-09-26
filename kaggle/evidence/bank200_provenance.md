# Bank provenance: the pre-registered 200-prompt calibration bank

Recorded from the merged bank on branch calib-200. The bank itself is git-ignored
(3.7 MB of per-prompt JSON); these hashes plus the producing kernels are the audit
trail, and every shard can be re-downloaded from its kernel output.

## Corpus

| Field | Value |
| --- | --- |
| prompts | 200 (prompt_id 0..199) |
| selection seed | 20260926 |
| split seed | 20260918 |
| folds | 5 of size 40 |
| source sha256 | 3381897271e4ec863a459f233de7cc7147c18f2dac092bd612b3649a92569b51 |
| geneval sha256 | 5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13 |
| exact GenEval overlap | 0 |
| candidates | 5000 (25 per prompt) |
| checkpoint scores per candidate | 13 + final |
| bank sha256 | 54a1fa41a5525718a285c5ef8035472571a30a6678419dc413b787cbb4160997 |
| fold agreement with manifest | True |
| candidate decode+reward time | 94.3 s/prompt |
| wall time per prompt | 323.1 s |
| total GPU time | 64629.3 s = 17.95 GPU-h |
| peak VRAM | 7.41 GiB |
| merge check | validate_bank.py --expected-prompts 200 --bank-dir bank200 (PASS) |
| numpy recorded | 1.26.4 |
| python recorded | 3.12.3 |

## Producing kernels (30 sessions, num_workers=60, 2x T4 each)

~~~text
codemaivanngu/tqk-bank200-260926-0826-0-1
codemaivanngu/tqk-bank200-260926-0826-2-3
kieutung/tqk-bank200-260926-0826-4-5
kieutung/tqk-bank200-260926-0826-6-7
bangchi/tqk-bank200-260926-0826-8-9
bangchi/tqk-bank200-260926-0826-10-11
iamlonely/tqk-bank200-260926-0826-12-13
iamlonely/tqk-bank200-260926-0826-14-15
anhhaphan/tqk-bank200-260926-0826-16-17
anhhaphan/tqk-bank200-260926-0826-18-19
ctlcmleon/tqk-bank200-260926-0826-20-21
ctlcmleon/tqk-bank200-260926-0826-22-23
damtrunghieu/tqk-bank200-260926-0826-24-25
damtrunghieu/tqk-bank200-260926-0826-26-27
hoanganpham123/tqk-bank200-260926-0826-28-29
hoanganpham123/tqk-bank200-260926-0826-30-31
johnntlhudson/tqk-bank200-260926-0826-32-33
johnntlhudson/tqk-bank200-260926-0826-34-35
kieuhongquan/tqk-bank200-260926-0826-36-37
kieuhongquan/tqk-bank200-260926-0826-38-39
manh1904/tqk-bank200-260926-0826-40-41
manh1904/tqk-bank200-260926-0826-42-43
nguyncmnhda/tqk-bank200-260926-0826-44-45
nguyncmnhda/tqk-bank200-260926-0826-46-47
phamdotuandng/tqk-bank200-260926-0826-48-49
phamdotuandng/tqk-bank200-260926-0826-50-51
veilwings/tqk-bank200-260926-0826-52-53
veilwings/tqk-bank200-260926-0826-54-55
victorharvey27/tqk-bank200-260926-0826-56-57
victorharvey27/tqk-bank200-260926-0826-58-59
~~~

Re-download: export KAGGLE_CONFIG_DIR for the kernel owner (see
scripts/kaggle_accounts.py), run
kaggle kernels output OWNER/SLUG -p DIR --file-pattern '.*bank200/.*[.]json$', then
kaggle/merge_shards.py --input DIR ... --bank-dir bank200 --expect-prompts 200.
