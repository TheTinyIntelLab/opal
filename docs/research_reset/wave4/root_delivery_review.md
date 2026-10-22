# Delivery review and numerical test portability

The serialized full suite is running from delivery-start.json's bound bytes.
Root found a portability issue by static review: the numerical no-overwrite
test uses the real Windows freeze manifest, whose relative keys contain
backslashes. The unchanged diagnostic's ROOT/name resolution is Windows-local,
so that test would reach a missing manifest path on Linux before checking the
existing output. B's source validator already normalizes its manifest paths.

Root will make this guard test use a complete synthetic authorized freeze with
no biological paths, rather than modifying or rerunning the successful N1
implementation. This isolates the behavior actually under test and avoids
claiming cross-platform real-state execution. The old test bytes and their
13-pass receipts remain preserved by hashes. A/C should leave diagnostic code
unchanged and include this test-fixture correction in final reporting. Root
will bind the revised fixture and execute its focused suite before delivery.

No further arithmetic candidate, scientific retry or real-state diagnostic
change is authorized. N3 still uses the original successful diagnostic bytes.
The legacy manifest's Windows-local path handling remains a disclosed tool
limitation rather than an accepted portable prediction adapter.
