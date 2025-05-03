# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build: `cargo build`
- Run: `cargo run --release`
- Check: `cargo check`
- Test: `cargo test`
- Test single test: `cargo test test_name`
- Format: `cargo fmt`
- Lint: `cargo clippy`

## Code Style Guidelines
- **Imports**: Group standard library imports first, then external crates, then local modules
- **Formatting**: Follow Rust standard formatting (rustfmt)
- **Types**: Use appropriate Rust types; prefer strong typing with structs over primitive types
- **Naming**: Use snake_case for variables/functions, CamelCase for types/traits
- **Error Handling**: Use Result<T, E> for operations that can fail; avoid unwrap() in production code
- **Comments**: Document public APIs with doc comments (///)
- **Methods**: Implement methods on structs using impl blocks
- **Vector Operations**: Use the Vec3 utility methods for vector operations