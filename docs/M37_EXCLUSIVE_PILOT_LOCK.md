# M37 — Exclusive guarded pilot execution lock

**Status:** Implemented for `pilot_train.py --execute`; no 125M weights
or accelerator-training evidence is shipped.

## Why

M36 checks disk capacity and the next checkpoint names, but two separate
launcher processes could check the same destination simultaneously and start
training together. M37 makes cooperating guarded launchers mutually exclusive
per output path, using atomic creation of a directory named
`.<checkpoint-filename>.aetherion-pilot.lock` alongside the output.

The lock is taken **after** corpus, hardware-report and live-device preflights,
and is held across M36 storage checks and the entire training child process.
The lock owner information is a local `owner.json` file containing PID,
creation timestamp, random run token and absolute output path. These fields
aid diagnosis; they are **not cryptographic authentication**.

## Failure policy

- If a lock directory already exists, execution fails closed.
- On normal return and Python exceptions, the launcher removes the lock only
  if the directory identity and owner token still match what it created.
- A crash, forced termination, power loss or machine restart can leave an
  orphaned lock. **It is deliberately not auto-reclaimed.**
- An operator must inspect whether the PID/job is still running, inspect
  checkpoints and verify no other writer is active before deleting a stale
  lock directory manually.
- If the parent directory does not yet exist, the launcher creates it for
  the lock. This is an intentional side effect limited to `--execute`.

## Boundaries

M37 protects only launchers following `pilot_train.py --execute` and using
the same resolved output parent and filename. An independent process,
`train.py` invoked directly, or a user forcibly removing a live lock can
still write to those paths. Locking does not reserve storage bytes, prevent
symlink/path races, authenticate host processes, guarantee an uninterrupted
training run, or ensure a checkpoint is valid.

Retained snapshots use the same base output filename for the shared lock.
A successful launch still requires M27/M28/M34/M35/M36 safety gates.

## Tests

```sh
python -m pytest -q tests/test_m37_pilot_lock.py
python -m pytest -q
```

Cases cover simultaneous attempts on the same destination, independent
destinations, relative/absolute aliases, owner metadata changes, exceptions,
stale locks, and simulated (never executed) training launches.
