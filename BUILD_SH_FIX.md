# Fix build.sh permissions for Render deployment

## Solution for the Git Permission Error

The error occurs because `build.sh` needs to be added to Git first before setting permissions. Here's the correct sequence:

### Step 1: Add build.sh to Git
```bash
git add build.sh
```

### Step 2: Make it executable
```bash
git update-index --chmod=+x build.sh
```

### Step 3: Commit the changes
```bash
git commit -m "Add build.sh with execute permissions"
```

## Alternative: Set permissions after commit

If the above doesn't work, you can also:

1. **Commit the file first:**
   ```bash
   git add build.sh
   git commit -m "Add build.sh"
   ```

2. **Then set permissions:**
   ```bash
   git update-index --chmod=+x build.sh
   git commit -m "Make build.sh executable"
   ```

## Important Note for Windows

On Windows, Git may warn about line endings (LF vs CRLF). This is normal. Render will handle Unix line endings correctly. The warning won't affect deployment.

## Verify it worked

After setting permissions, verify with:
```bash
git ls-files --stage build.sh
```

You should see `100755` in the mode column (executable) instead of `100644` (regular file).

## If Still Having Issues

Render.com will also work if you:
1. Ensure `build.sh` has `#!/bin/bash` as the first line (it does)
2. Make sure it's committed to the repository
3. Render will execute it during the build process

The `render.yaml` file specifies `buildCommand: ./build.sh`, so Render will run it automatically.

