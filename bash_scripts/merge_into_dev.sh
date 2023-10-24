#!/bin/bash

# Fetch the latest changes from the remote
git fetch -p origin

# Get the name of the current branch to return to it at the end
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Stash any changes in the working directory and stage
git stash push -u -m "Temporary stash for merge_into_dev script"

# Ensure we're working with the latest version of dev
git checkout dev
git pull origin dev

# For each branch, except master and dev
for branch in $(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v -e '^master$' -e '^dev$' -e '^notebooks$'); do
    echo "Merging branch $branch into dev..."

    # Attempt to merge the branch into dev
    git merge $branch --no-edit

    # If there are merge conflicts
    if [ $? -ne 0 ]; then
        echo "Merge conflict when merging $branch into dev. Skipping..."
        # Abort the merge and move to the next branch
        git merge --abort
    else
        echo "Merged $branch into dev successfully."
    fi
done

# Switch back to the original branch
git checkout $current_branch

# Pop the stashed changes to restore the working directory and stage
if git stash list | grep -q "Temporary stash for merge_into_dev script"; then
    git stash pop "$(git stash list | grep "Temporary stash for merge_into_dev script" | awk -F: '{print $1}')"
fi

echo "Done."
