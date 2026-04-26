"""Fix all open() calls in audit.py to use utf-8 encoding."""
content = open('audit.py', encoding='utf-8').read()

files_to_fix = [
    'cuad_loader.py', 'tier1_grader.py', 'tier1_runner.py',
    'tier3_environment.py', 'lexarena_runner.py', 'lexarena_scorer.py',
    'lexarena_server.py', 'lexarena_report.py', 'lexarena_benchmark.py',
    'leaderboard/index.html', 'paper/lexarena_paper.tex', 'paper/lexarena.bib',
    'probe_runner.py', 'data/manifest.json',
]
for fn in files_to_fix:
    old = f'open("{fn}")'
    new = f'open("{fn}", encoding="utf-8")'
    content = content.replace(old, new)

# Also fix open("data/manifest.json") pattern used in json.load
open('audit.py', 'w', encoding='utf-8').write(content)
print('Done — fixed', len(files_to_fix), 'open() calls')
