# Contributing

Changes should be focused, documented, and accompanied by regression tests for
numerical behavior. Before opening a pull request, run:

```sh
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Public API changes must preserve validated model invariants. Mathematical
changes should state the modeled intensity, likelihood, and stationarity
condition in the pull request.
