# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Report it privately to the maintainers (use the
repository's "Report a vulnerability" / security advisory feature, or contact the project owner directly) with steps to
reproduce. We aim to acknowledge reports within a week.

## What to keep in mind when you deploy this project

* **Secrets live in `backend/.env`**, which is git-ignored. Never commit it. Set a strong `JWT_SECRET` (32+ characters);
  without one the server generates a random secret per run and logs users out on every restart.
* **CORS** only allows the origins in `CORS_ORIGINS` (default: the local Expo web app).
* **Uploaded documents** are saved as extracted text in the operating system's temp folder (`annotationtool_docs`).
  They are deleted when the user removes them, and files older than 6 hours are cleaned up the next time anyone uploads
  a document (there is no background timer). Do not run the server on a shared machine with sensitive documents unless
  that is acceptable.
* **Offline models run third-party code.** The IndicTrans2 models are loaded with `trust_remote_code`, which executes
  Python shipped with the model. The exact commits are pinned in `backend/local_translator.py`; review any change to those
  pins. Set `LOCAL_INDICTRANS=0` to refuse this entirely.
* **Cloud translation sends your text to a third party** (Azure, Google or MyMemory) when enabled. The default
  `local` provider keeps text on your machine.
* The server is built for single-machine use: jobs are kept in memory and uploaded files on local disk. Running several
  server processes behind a load balancer needs shared storage and a real job queue.
