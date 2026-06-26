This directory holds local checkouts of the Rust-backed search dependencies used
by the profiling branch:

- `tachiom`
- `xtr-warp-rs`
- `fast-plaid`

The checkout contents are ignored by PyLate git. In this workspace they are
symlinked to:

```text
/exp/rjha/tachiom
/exp/rjha/xtr-warp-rs
/exp/rjha/fast-plaid
```

`pyproject.toml` points `uv` at these paths through `[tool.uv.sources]`, so
`uv run` and `uv sync` use local editable builds instead of released wheels.

Push profiling changes to the user forks, not upstream:

- `tachiom`: remote `fork` (`https://github.com/robro612/tachiom.git`)
- `fast-plaid`: remote `origin` (`https://github.com/robro612/fast-plaid.git`)
- `xtr-warp-rs`: remote `origin` (`https://github.com/robro612/xtr-warp-rs.git`)

As of this profiling pass, Tachiom and FastPlaid have committed `tracing`
bridge/search-span changes. WARP / xtr-warp-rs is wired as a local dependency
but its rust-internal tracing is still pending.

To recreate the local layout in another checkout:

```bash
mkdir -p third_party
ln -s /path/to/tachiom third_party/tachiom
ln -s /path/to/xtr-warp-rs third_party/xtr-warp-rs
ln -s /path/to/fast-plaid third_party/fast-plaid
```
