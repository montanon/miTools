#!/bin/bash

# Fetch the latest changes from the remote to ensure we're up to date
git fetch -p origin

# Update dev to make sure it's up to the latest commit
git checkout dev
git pull origin dev

# Get the name of the current branch to return to it at the end
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Stash any changes in the working directory and stage
git stash push -u -m "Temporary stash for delete_branches script"

# For each branch, except master and dev
for branch in $(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v -e '^master$' -e '^dev$' -e '^notebooks$'); do
    # Find the common ancestor of the branch and dev
    merge_base=$(git merge-base $branch dev)
    # Check if the branch is identical to dev
     if [ $(git rev-parse $branch) == $merge_base ]; then
        # Delete the branch
        git branch -d $branch
        echo "Deleted branch $branch as it was up to date with dev."
    else
        echo "Branch $branch is not up to date with dev. Skipping."
    fi
done

# Switch back to the original branch
git checkout $current_branch

# Pop the stashed changes to restore the working directory and stage
if git stash list | grep -q "Temporary stash for delete_branches script"; then
    git stash pop "$(git stash list | grep "Temporary stash for delete_branches script" | awk -F: '{print $1}')"
fi

echo "Done."
