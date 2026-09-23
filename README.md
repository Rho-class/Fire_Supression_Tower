## Branching Strategy

We use feature branches so multiple team members can work simultaneously while keeping `main` stable.

### Branches

- `main` - Stable, working version of the project
- `feature/...` - Individual features or tasks

Examples:
- `feature/fire-detection`
- `feature/fire-localization`
- `feature/ir-sensor`
- `feature/motor-control`
- `feature/pump-control`

### Workflow

1. Get the latest version of main: `git checkout main` → `git pull origin main`
2. Create your feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git add .` → `git commit -m "Describe your changes"`
4. Push your branch to GitHub: `git push -u origin feature/your-feature-name`
5. Open a Pull Request on GitHub: `feature/your-feature-name` → `main`
6. After your feature is merged: `git checkout main` → `git pull origin main`

### Multiple People on the Same Feature

If multiple people are working on the same feature, split the work into smaller feature branches.

Example:

    main
    ├── feature/fire-localization
    ├── feature/ir-sensor
    ├── feature/motor-control
    └── feature/pump-control

Do not push unfinished code directly to `main`. Always create a feature branch and use a Pull Request.