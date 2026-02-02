# Palette's Journal - Petals UX/Accessibility Learnings

## 2025-05-15 - Initial exploration
**Learning:** Petals is primarily a CLI-based tool for hosting and using LLMs in a distributed network. UX improvements should focus on the command-line interface, providing clear feedback, helpful error messages, and better accessibility in the terminal.
**Action:** Focus on `src/petals/cli/` for potential micro-UX enhancements.

## 2025-05-15 - CLI Argument Consolidation and Validation
**Learning:** Consolidating value-based arguments and flags (like `--token` and `--use_auth_token`) into a single `nargs='?'` argument improves CLI ergonomics while maintaining backward compatibility. Replacing `assert` statements with `parser.error()` is critical for a "pleasant" CLI experience, as it avoids leaking implementation details (tracebacks) to the user.
**Action:** Always prefer `parser.error()` for user-facing validation in Python CLI tools. Use `nargs='?', const=True` to allow optional values for flags.
