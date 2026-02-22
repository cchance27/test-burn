The following are design goals and rules for our Agents and Developers for the project. 

- Our top priority is PERFORMANCE and LATENCY, followed closely by Memory usage. 
- Developer Experience is also critical concern and we should take the effort to consider clean DX when adding features.
- Practice defensive programming whenever we possibly can, fail early and fail hard where it makes sense.
- Refactor when absolutely needed, as we attempt to maintain good code cleanliness, including SOLID and DRY principals, taking care to follow SRP.
- Refactor out duplicate code where possible into small primatives that can be reused and optimized and tested. 
- unsafe code should have safe wrappers and we should only use our safe wrappers, the unsafe code should be safety documented and have defensive coded checks where possible.
- All major features should have comprehensive tests written and stored in the metallic::tests module space. 
- Perform cargo build runs periodically to make sure to avoid regressions.
- Do not roll back code or delete files without checking with a superior. 
-  The project is stored on git commits, so you can reference git diffs and status when you want to review recent changes.
- Before we perform a git commit (when requested by the user), perform final cleanup checks.
   a. Run `cargo +nightly fmt` && `cargo clippy --fix --allow-dirty --allow-staged` && `cargo build`
- NEVER return a task or mark it as complete with PLACEHOLDER functions are placeholder comments, we implement functions and components fully. Only ever use placeholders if we plan to replace the placeholder in our next step that we're already planning to execute, and placeholders should have todo!() so that it will crash out if we forget to finish them.
- Always use idiomatic rust where possible.
- Don't force tests to pass ever, tests should be properly setup and logical, if a test fails it either has a mistake OR its showing a valid issue in our code that should be researched and fixed correctly.
- Make sure we use strongly typed errors whenever possible (avoid unwraps and expect outside of tests)
- Remember performance is #1 priority, so cloning should be minimized we should try to use zero copy abstractions where possible. 
- Use enums where possible so that we can rely on exhaustive and nonexhaustive match cases to avoid footguns.
- Note in comments tech debt with obvious DEBT: designations, but try to avoid tech debt that we will have to deal with later.
- Use the fastest possible primatives you can don't use HashMap if we can use FxHashMap which is known to be faster
- Tests should be written to check for semantic issues, as well as under-and-over runs for extrme values, and other items.
- Unit tests should test functionality to confirm base functionality is as expected.
- Unit tests should have strict tollerances and we should avoid adjusting tests to work around issues, and instead fix issues if the tests are valid.
- Please remember to context.synchronize() as needed to make sure that tensors are settled in the gpu memory when created or used.
- If updating code that has comments that reference it make sure the comments are updated to match the new changes.
- Use ideomatic rust always
- Always use `cargo add xxxx` to add packages don't manually edit cargo files.
- When reviewing json files, use jq, grep and other bash commands, if more advanced parsing is needed use a python one liner, as json files could be extremely large so cating them or reading those files directly could be a major issue.
- When reviewing logs or csv files use bash commands (grep, awk, sed, cut, sort etc), as the files may be very large so reading them or catting them is ill advised. 
- When executing cargo commands make sure to use options like -q and --message-format=short to minimize output and maximize context usage.

## Maintainability Rules (New Baseline)

- Maintainability work must preserve PERFORMANCE/LATENCY first, then memory usage, while improving DX.
- Baseline snapshot (2026-02-22): `272` `.rs` files, `34` files over `500` lines, `7` files over `1000` lines, and `50` files with inline `#[cfg(test)]` or `mod tests`.
- Line-count targets for Rust source files:
  - Line-count limits apply to production/source modules, not dedicated test files (`*.test.rs` or crate `tests/` trees).
  - Soft limit: `500` lines per `.rs` file.
  - Warning threshold: `700` lines; new code should be split unless there is a documented performance reason.
  - Hard review threshold: `1000` lines; splitting is required unless explicitly approved with rationale.
- Prefer module decomposition over monolith files:
  - Use directory modules with `mod.rs` and focused sub-files by responsibility.
  - Keep internal details private (`pub(crate)` or private) and expose stable surface area via curated `pub use` in `mod.rs`.
  - Avoid interface leakage across domains; only export what downstream code needs.
- Unit test file organization standard:
  - Prefer separate unit test files next to implementation: `something.rs` and `something.test.rs`.
  - Parent module should include test file with path mapping, for example: `#[path = "something.test.rs"] mod something_test;`.
  - Test files should use crate-level gate `#![cfg(test)]` to avoid scattering `#[cfg(test)]` across imports/items.
  - Do not duplicate test gating: if `something.test.rs` has `#![cfg(test)]`, do not also add `#[cfg(test)]` at the include callsite.
  - Keep integration tests in crate `tests/` directories; unit tests stay colocated with the module they validate.
- Environment variable access standard:
  - Use `metallic-env` for all environment reads/writes in crate code (typed descriptors and helpers like `is_set`).
  - Avoid direct `std::env::{var,var_os,set_var,remove_var}` in production code except inside `metallic-env` itself.
- Refactoring policy for maintainability:
  - During feature work, if a touched file is already above `500` lines, opportunistically split by SRP boundaries.
  - Eliminate obvious duplication by extracting small reusable primitives.
  - Keep behavior identical unless change is intentional and covered by tests.
