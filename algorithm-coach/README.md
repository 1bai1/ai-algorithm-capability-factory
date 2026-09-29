# Algorithm Coach

Algorithm Coach is the business layer built on top of the Pi coding-agent framework.
It targets algorithm programming education: understand a problem, retrieve relevant
algorithm knowledge, generate a runnable solution, validate it, repair failures, and
store the result as reusable experience.

## Relationship to Pi

`..\pi-main` provides the agent loop, LLM access, session management, and built-in
tools. Both directories are committed so a fresh clone includes the full runtime.

## Directory layout

```text
knowledge/                 Algorithm concepts and solution patterns
memory/experiences/        Successful and failed validation records
tasks/                     Problem statements and test cases
solutions/                 Generated solution files
reports/                   Unified validation reports
harness/                   Isolated code execution and evaluation
.pi/skills/algorithm-coach Pi workflow instructions
```

## Intended workflow

```text
problem description -> knowledge retrieval -> plan -> code -> validation
-> repair from failure feedback -> experience write-back
```

The first milestone is a CLI demonstration for one Python algorithm problem. The
validator will execute generated code in an isolated temporary directory with a
timeout and emit a JSON report.

## Start Pi in this project

```powershell
$env:OPENCODE_API_KEY = "<your-key>"
Set-Location "algorithm-coach"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "..\pi-main\pi-test.ps1" `
  --model opencode-go/deepseek-v4.1-flash `
  --thinking max `
  --tools read,powershell,edit,write,grep,find,ls
```
