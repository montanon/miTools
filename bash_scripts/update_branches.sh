#!/bin/bash

# Fetch latest changes from remote
git fetch -p origin

# Get the name of the current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Stash any changes in the working directory and stage
git stash push -u -m "Temporary stash for update_branches script"

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

# Pop the stashed changes to restore the working directory and stage
if git stash list | grep -q "Temporary stash for update_branches script"; then
    git stash pop "$(git stash list | grep "Temporary stash for update_branches script" | awk -F: '{print $1}')"
fi

echo "Done."
