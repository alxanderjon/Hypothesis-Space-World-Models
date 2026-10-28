# Common-4D interface x mechanism factorial v1 — additive recovery 1

Status: prospective before recovery work.

The original attempt completed all four fits and 2,000 updates, then stopped
before persisting any prediction because the blind-control tensor left the
`no_grad` block before conversion to NumPy. The original namespace, plan,
checkpoints and terminal error remain immutable.

Recovery loads the four saved checkpoints. It deterministically reconstructs
the sealed synthetic laws and banks, verifies all checkpoint and plan hashes,
then repeats only data construction and prediction/scoring. It performs zero
fits, gradient calls, updates or model selection.

Recovery limits: 9,984 repeated RHS rows, 6,144 repeated transform rows, 32,768
new evaluation rows and 512 score blocks; 120 seconds, 1 GiB RSS and 32 MiB
artifacts. Cumulative accounting after successful recovery is 19,968 RHS rows,
12,288 transform rows, four fits, 2,000 updates, 35,840 attempted evaluation
rows (including the original unsaved 3,072) and 512 score blocks.

