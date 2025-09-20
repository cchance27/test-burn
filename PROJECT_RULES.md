The following are design goals and rules for our Agents and Developers for the project. 

1. Our top priority is PERFORMANCE and LATENCY, followed closely by Memory usage. 
2. Developer Experience is also critical concern and we should take the effort to consider clean DX when adding features.
3. Practice defensive programming whenever we possibly can, fail early and fail hard where it makes sense.
4. Refactor when absolutely needed, as we attempt to maintain good code cleanliness, including SOLID and DRY principals. 
5. Refactor out duplicate code where possible into small primatives that can be reused and optimized and tested. 
6. unsafe code should have safe wrappers and we should only use our safe wrappers, the unsafe code should be safety documented and have defensive coded checks where possible.
7. All major features should have comprehensive tests written and stored in the crate::metallic::tests module space. 
8. Perform cargo build runs periodically to make sure to avoid regressions.
9. Do not roll back code or delete files without checking with a superior. 
10. The project is stored on git commits, so you can reference git diffs and status when you want to review recent changes.
11. Before turning in a task as complete make sure that critical cargo commands were run.
   a. Run `cargo fmt` && `cargo clippy --fix --allow-dirty --allow-staged` && `cargo build`
12. NEVER return a task or mark it as complete with PLACEHOLDER functions are placeholder comments, we implement functions and components fully. Only ever use placeholders if we plan to replace the placeholder in our next step that we're already planning to execute, and placeholders should have todo!() so that it will crash out if we forget to finish them.
13. Always use idiomatic rust where possible.
14. When creating test and validating our code and functions, we should use pytorch or numpy for 1 off experiments or to generate expected outputs, for more complex tests that need large comparison data we can use burn-rs as a comparison tool (only in tests)
15. Don't force tests to pass ever, tests should be properly setup and logical, if a test fails it either has a mistake OR its showing a valid issue in our code that should be researched and fixed correctly.
16. Make sure we use strongly typed errors whenever possible (avoid unwraps and expect outside of tests)
17. Remember performance is #1 priority, so cloning should be minimized we should try to use zero copy abstractions where possible. 
18. Use enums where possible so that we can rely on exhaustive and nonexhaustive match cases to avoid footguns.
19. Note in comments tech debt with obvious DEBT: designations, but try to avoid tech debt that we will have to deal with later.