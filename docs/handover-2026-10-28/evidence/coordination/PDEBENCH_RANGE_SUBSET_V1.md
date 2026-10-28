# PDEBench range subset v1

Status: prospective before acquisition.

Official source commit:
`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`.

The selected official HDF5 files are uncompressed contiguous arrays with
10,000 trajectories and 1,024 spatial points.  Advection and Burgers contain
201 saved field states; reaction-diffusion contains 101.  Full single shards
are 8.233 GB, 8.233 GB and 4.137 GB respectively, so this pilot uses HTTP byte
ranges and never downloads a full shard.

Selected settings, fixed from the official metadata before field values:

- source advection: beta 0.2, 1.0, 4.0;
- source Burgers: viscosity 0.004, 0.04, 0.4;
- whole-mechanism development reaction-diffusion: `(nu,rho)` = `(0.5,1)`,
  `(1,2)`, `(5,10)`.

For every file, SHA-256 of `pdebench-range-subset-v1:<name>` seeds a local RNG
that selects 30 distinct trajectory indices from 0..9999.  The first 20 sorted
indices are acquisition-training, the next five development, and the final
five reserved and **not downloaded**.  This order is fixed without reading
field values.  No official test file is opened.

Each downloaded trajectory is read as one contiguous byte range.  Advection
and Burgers retain time indices `0,2,...,200`; reaction-diffusion retains
`0..100`, producing 101 samples without interpolation.  All retain spatial
indices `0,16,...,1008`, producing 64 points.  Original physical `x` and `t`
coordinates are stored.  No normalization, field clipping or learner split is
computed during acquisition.

Prospective limits: nine metadata inspections, 225 trajectory range reads,
at most 170 MiB network payload, 32 MiB persisted compact arrays, 600 seconds,
1 GiB RSS.  Each output records remote URL/file size/ETag, official full-file
MD5 from repository metadata, selected and reserved indices, original and
retained shapes, coordinate arrays and actual transferred bytes.  A partial
download cannot verify the official full-file MD5 and must not claim it.

This dataset remains a strong scalar-grid assumption.  It does not establish
arbitrary mechanism discovery, and the reaction-diffusion mechanism becomes
opened development evidence as soon as its five development trajectories are
downloaded.  Diffusion-sorption is the currently named but unopened candidate
whole-mechanism confirmation; it is not acquired by this protocol.
