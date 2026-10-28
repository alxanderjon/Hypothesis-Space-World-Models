# Smoke-screen artifact workflow

For new or updated smoke-screen training scripts:

1. Immediately after the final training update, save the weights with status
   `TRAINED_PENDING_EVALUATION`, before any geometry, validation, downstream,
   plotting, or acceptance-gate work. Use `smoke_artifacts.save_checkpoint` to
   write atomically, check reloaded weights, and emit a SHA256 sidecar.
2. Include seed, initialization recipe, training-data identifiers or generation
   recipe, optimizer/settings, update count, history, and implementation hash.
3. Save generated point sets and their source/state identifiers before evaluating
   them. Record their hashes. Keep these artifacts even if later gates fail.
4. Load persisted artifacts for downstream evaluation. Recovery/resume must not
   silently repeat training. Replaying training requires authorization when the
   current task prohibits training.
5. Save a separate `CANDIDATE` checkpoint only after the requested gates pass.
   Persistence alone is not promotion, locking, or production replacement.
6. Preserve existing locked files and hashes. Do not retrofit locked scripts in
   place. Verify protected hashes after the run.

The space-filling smoke script follows this workflow. Its recovery script also
persists replay weights before checking reproduction, and promotes them to a
verified recovery artifact only after the saved metrics match.
