# FRIDAY-v2: notes for Claude

## Git rules (strict)
- NEVER commit, push, or change git history. Do not run `git commit`, `git push`, `git add`, `git stash`, `git reset`, `git checkout`, `git rebase`, `git merge`, `git tag`, or `gh pr create`.
- Read-only git commands (`git status`, `git diff`, `git log`) are fine.
- Never add "Co-Authored-By" lines or any Claude/Anthropic attribution anywhere.
- The owner reviews and commits all changes himself. When work is done, stop and summarize the changes instead.

## Project
- Local voice assistant running on an NVIDIA Jetson: Whisper (speech-to-text), Ollama qwen2.5:0.5b (LLM), Piper (text-to-speech), OpenCV YuNet/SFace (face recognition), an Arduino over serial (/dev/ttyACM0), and a laptop that receives commands over HTTP.
- Hardware isn't available in Claude's environment. Verify changes with `python -m py_compile` and static checks, and say what couldn't be tested.
- Don't modify faces.json or memory.json; they hold personal data.
