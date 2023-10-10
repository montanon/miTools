#!/bin/bash

# Fetch the latest changes from the remote
git fetch -p origin

# Get the name of the current branch to return to it at the end
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Stash any changes in the working directory and stage
git stash push -u -m "Temporary stash for push_branches script"

# For each branch, except master and dev
for branch in $(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v -e '^master$' -e '^dev$'); do
    echo "Pushing branch $branch to origin..."

    # Attempt to push the branch to the remote
    git push origin $branch

    # If there are any errors in pushing
    if [ $? -ne 0 ]; then
        echo "Error pushing branch $branch to origin. Skipping..."
    else
        echo "Pushed $branch to origin successfully."
    fi
done

# Switch back to the original branch
git checkout $current_branch

# Pop the stashed changes to restore the working directory and stage
if git stash list | grep -q "Temporary stash for push_branches script"; then
    git stash pop "$(git stash list | grep "Temporary stash for push_branches script" | awk -F: '{print $1}')"
fi

echo "Done."
