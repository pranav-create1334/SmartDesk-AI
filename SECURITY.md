# Security Policy

## Scope

SmartDesk AI is a local academic demonstration project. It does not call paid
external APIs and does not require cloud credentials. The Streamlit application,
SQLite database and Machine Learning model all run on the local machine.

## Secrets and sensitive data

This repository must never contain:

- `.env` files or environment-specific credentials
- `.streamlit/secrets.toml`
- API keys, access tokens, private keys or passwords
- Real user complaints, names, email addresses or other personal data
- The local SQLite database (`database/helpdesk.db`)
- The local Python virtual environment (`venv/`)
- Generated model binaries and evaluation metrics

The `.gitignore` file excludes these files. If any sensitive file is
accidentally committed:

1. Revoke or rotate the exposed credential immediately.
2. Remove the file from Git tracking:
   `git rm --cached <file>`
3. Purge it from Git history before publishing the repository.
4. Treat the credential as compromised even after it is deleted.

The included complaint dataset is synthetic and written for demonstration. It
does not contain real personal information.

If credentials are added in the future, provide them through environment
variables or Streamlit secrets locally, keep a placeholder-only
`.env.example`, and never commit the real values.

## Reporting a vulnerability

Please do not create a public issue for a security problem. Use GitHub's
private vulnerability reporting or contact the repository owner through their
GitHub profile.

## Dependency safety

Runtime dependencies are pinned in `requirements.txt`. Before deploying, review
them with `python -m pip list --outdated` and a dependency vulnerability scanner.
This project is intended for local demonstration, not production deployment.

## Known security limitations

- Administrator pages do not require authentication.
- The application is designed to bind to a local Streamlit server.
- SQLite is not intended for high-concurrency production use.
- Complaints should be anonymised before being used in any future real dataset.
