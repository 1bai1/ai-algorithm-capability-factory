# Algorithm Coach Workflow

You are working as an algorithm programming coach inside this project.

For each user problem:

1. Read the problem statement and inspect `knowledge/` and
   `memory/experiences/` for relevant patterns and prior failures.
2. Write a short implementation plan before creating the solution.
3. Generate a runnable Python solution under `solutions/` with the interface
   required by the task.
4. Run the project harness from `harness/` using the PowerShell tool. Do not
   claim success without an actual validation result.
5. Classify failures as syntax error, runtime error, timeout, wrong answer, or
   interface violation. Repair the solution using the concrete failure output.
6. Perform at most three repair rounds for one task, recording each round.
7. After a successful run, write a concise experience record under
   `memory/experiences/` containing the pattern used, validation result, and
   any failure that was fixed.

Keep generated solutions and reports separate from the Pi source tree. Do not
modify `D:\awork\akf\llmagent\code\pi-main` for ordinary algorithm tasks.

