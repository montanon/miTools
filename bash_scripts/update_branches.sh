#!/bin/bash

# Fetch latest changes from remote
git fetch origin

# Get the name of the current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# For each branch, except master
for branch in $(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v '^master$'); do
    echo "Checking branch $branch..."
    
    # Switch to the branch
    git checkout $branch

    # Merge dev into the branch
    git merge origin/dev --no-edit

    # If there are merge conflicts
    if [ $? -ne 0 ]; then
        echo "Merge conflict in branch $branch. Skipping..."
        # Abort the merge and move to the next branch
        git merge --abort
    else
        echo "Merged dev into $branch successfully."
    fi
done

# Switch back to the original branch
git checkout $current_branch

echo "Done."
