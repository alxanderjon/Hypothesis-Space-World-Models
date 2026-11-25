# Hypothesis-Space World Models: goal and standing rules

Every session working in this repository reads this file first. It states the end goal, the one definition of "solved",
and the standing rules the owner has set. The research log (`docs/research-log.md`) records how each was reached.

## The end goal

A learner that recovers dynamical systems from data **so exactly that its equations reproduce the true trajectories over
a long horizon from new starting states**, and that **learns new systems with fewer tries as its experience grows**.

Both halves count:
- **Exactness:** what is recovered must be the system's law, not a curve fit to the recordings.
- **Accumulation:** with experience, new systems are recovered with fewer search tries and more often than by a learner
  without that experience. Retained knowledge must not be lost (no forgetting).

## What "solved" means (the only definition)

A system counts as **solved** only if all of the following hold.

1. **Recording checks.** The found equations, refitted, match the recorded step changes, integrated across each recorded
   step from the recorded states:
   - relative error at most **max(1e-9, 30 × the equation's noise floor)**;
   - on the fitting recordings and on held-back recordings;
   - no term degenerate on the data.
2. **High-precision fit.** After a structure is accepted, all its coefficients and constants are refitted together by
   shooting: the model is integrated over each recorded sample interval and matched to the next recorded state.
3. **The reference.** For each system, the **true structure** goes through the same recording checks and high-precision
   fit, **from the same recordings the learner saw**. Its errors (items 4–5) set the system's bars: the recordings, in
   float64, can pin the constants only so far, and no learner can be asked to beat the true law refitted from them.
4. **One step from new starting states.** The found right-hand side is compared with the exact true one:
   - at **20 new starting states** (10 inside the recorded range, 10 at 1.5–2× its half-width outside it), and
   - at **2,000 states** of the same region (half inside, half at 1–2× the half-width outside).

   On each set, its largest relative error must be at most **max(30 × the reference's, 1e-12)**.
5. **Rollouts.** From the 20 starts, the found system's trajectories over **10 of the system's time constants** must stay
   within **max(30 × the reference's error, 1e-9)** (relative) of the true trajectories.
6. **Unjudgeable systems.** A system cannot be judged if the reference fails the recording checks, if the reference's
   one-step (20 starts or 2,000 states) or rollout error exceeds 1e-6, or if a raised bar exceeds **1e-8 one step** or
   **1e-5 rollout**. New rungs replace such systems; retroactive checks exclude them and report how many.
7. **Bounded systems.** Training and test systems are drawn so that the true system stays bounded over that horizon from
   all 20 test starting states. Systems that blow up are rejected by the generator.
8. **The valid range.** Systems are judged on the states they can actually take. A variable whose recordings never change
   sign keeps its sign in the test starts and the 2,000 states, so an equation that differs from the truth only at
   unreachable values (for example |x| for an always-positive x) counts as correct.

**Why these numbers** (calibrated on training systems, 24–25 November; research log): refitting the true structure from
starting constants kicked by 1e-6 shows how far a correct answer can land from the reference by chance. The margin 30
rejects 11 of 474 such correct refits (10x rejected 19). The floors (1e-12, 1e-9) mean the rule is never stricter than
the absolute bars. The ceilings stop a good approximation from passing as exact on systems the recordings pin poorly.

**Implementation.** `experiments/accumulating_learner/rollout_check.py` (with `precise_fit.py` and `fitter_g1.py`) judges
held-out systems, building the reference and reporting the absolute verdict alongside; `gen_wide.py` draws bounded
systems from wide recordings.
- Recording spread: 0.2–3× the initial state at 2–3 variables, 0.3–2× at 4.
- Each system's description, test starting states and horizon are stored when it is generated.
- Dreams (practice systems whose true answers are known) may use the cheaper generator.

**Systems drawn before 23 November 2026** (narrow recordings, not checked for boundedness) are judged retroactively with
each test start counted while the true system stays bounded, for at least one time constant.

## Proxies must be named as proxies

Every new task, test, benchmark or pass rule must **state how it measures the end goal**.

Any proxy must be **flagged as a proxy** and **never reported as "solved"**, for example:
- recordings-only exactness;
- ranks of true terms;
- top-10 shares;
- fit error;
- symbolic look-alike labels (comparing forms: they flag equations equal to the truth on the valid range).

(Until 22 November 2026 the ladder's "solved" was recordings-only. That was a proxy, and it passed look-alike equations
that were wrong off the recordings.)

## Standing rules

**Method:**
- **One change at a time.** Each rung or test changes one thing, so a failure can be located.
- **Minutes-scale checks before long runs.** A quick check (no search where possible) comes before any hours-scale test.
  It measures cost (time, compute, disk) as well as benefit.
- **Pass rules are written into the log before the run.**
- **Several seeds.** At least 3 seeds per arm for every comparison. A result near p = 0.01–0.05 is confirmed by a rerun.
  Reruns are not bitwise identical.
- **Controls.** A fresh network (no history) in every comparison; made-up systems as a control for any transfer claim.
- **Equal training across arms.** Every arm gets the same number of training updates per round.
- **Reanalyse in every run.** After each learning round, unsolved training systems are re-searched with the newest network.
- **Mastery and drop-off.** A rung is mastered when an arm solves at least 8 of 10 held-out systems in at least 2 of 3
  seeds. If neither learner arm masters it, stop and report.
- **Mastered means mastered under the real definition.** A rung counts as mastered only under the definition of solved
  above, and **no rung is attempted until every rung below it is mastered that way.** New rungs replace unjudgeable
  held-out systems (always 10 judgeable); retroactive checks count mastery as at least 80% of the judgeable held-out
  systems and report the unjudgeable count.
- **Generous search budgets when learning** (thousands of tries per equation); report tries at 300 as well, for
  comparison with early rungs.
- **The learning measure.** Tries per solution must fall, compared on the same held-out equations solved in both the first
  and the last round. A fixed probe set of held-out systems from earlier rungs measures accumulation and forgetting across
  rungs.

**What the learner may use:**
- **Teacher dreams are approved teaching.** Teacher-made practice systems are allowed. The no-hand-added-pieces rule
  concerns the learner's toolkit (language, library, search), not its practice problems.
- **No hand-added pieces in the toolkit.** Pieces are discovered from data and reused, never added by hand.
- **New pieces are introduced in the easy setting first** (2 variables, 1–2 terms per equation), mastered there, and only
  then used in longer equations or with more variables.
- **LLM help is a tool, not a shortcut.** Every LLM suggestion is verified by the exactness check. Its benefit is measured
  only on systems it cannot have memorized (newly generated or unpublished).

**Diagnosis:**
- **Compare against mastered cases before diagnosing a failure.** A property seen in a failure (for example, ambiguous
  evidence) is a cause only if the mastered cases do not share it.
- **Check freshness and protect held-out data.** New systems are checked against every earlier set; only counts of test
  data are printed; reserved sets stay untouched.

**The shared server (ironforge):**
- **Use the server's resources fully** (GPUs, CPU cores, memory, both disks), sized to what is actually free.
- **Safety guards:**
  - stop writing if any disk in use falls below 2 GB free;
  - never train on the CPU; one training per GPU.
- **Never interfere with other users' processes.** Stop only your own processes, by process id or process group.
- **Never modify a module that a running job imports.** Running jobs reload code when they start new worker processes.

**Reporting:**
- **Plain language.** Reports say what was tested, the deciding number, and pass or fail.
- **Faithful.** A failure is reported as a failure, with what was tried. A proxy is reported as a proxy.
- **Check in before every new trial; record the owner's decisions in the log.**

**Repository hygiene:**
- Never commit `src/hypothesis_space_world_models.egg-info/PKG-INFO` or
  `experiments/sho/run_stage1_per_trajectory_omega_curriculum.py`.
- Stage files by name.
