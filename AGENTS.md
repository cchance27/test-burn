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
-  Before turning in a task as complete make sure that critical cargo commands were run.
   a. Run `cargo fmt` && `cargo clippy --fix --allow-dirty --allow-staged` && `cargo build`
- NEVER return a task or mark it as complete with PLACEHOLDER functions are placeholder comments, we implement functions and components fully. Only ever use placeholders if we plan to replace the placeholder in our next step that we're already planning to execute, and placeholders should have todo!() so that it will crash out if we forget to finish them.
- Always use idiomatic rust where possible.
- When creating test and validating our code and functions, we should use pytorch or numpy for 1 off experiments or to generate expected outputs, for more complex tests that need large comparison data we can use burn-rs as a comparison tool (only in tests)
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
- DO NOT edit cargo.toml and cargo.lock, to add or modify crates use cargo add/remove etc.