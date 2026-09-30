# Validation Harness

The harness will execute a generated Python solution in a temporary working
directory, enforce a timeout, run public and hidden tests, and emit one JSON
report into the task directory under `knowledge/任务池/<任务>/validation/`.

The report must distinguish syntax errors, runtime errors, timeouts, wrong
answers, and interface violations so the Agent can repair from structured
feedback.

