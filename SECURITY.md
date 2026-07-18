# Security policy

Please report security or integrity issues privately through GitHub's security
advisory interface for this repository. Include a minimal reproducer, affected
versions, and the expected impact. Do not include credentials, proprietary
datasets, or personal data in reports.

The crate performs numerical modeling and is not a security boundary. Invalid
or adversarial floating-point inputs should still return an error rather than
corrupt model invariants; reports of violations are welcome.
