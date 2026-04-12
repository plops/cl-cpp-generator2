#!/bin/bash


# Function to format a single file
format_lisp_file() {
    local file=$1
    echo "Formatting $file with emacs..."

    emacs --batch -l ~/.emacs "$file" \
           --eval "(package-initialize)" \
           --eval "(require 'slime)" \
           --eval "(slime-setup '(slime-cl-indent))" \
           --eval "(setq lisp-indent-function 'common-lisp-indent-function)" \
           --eval "(indent-region (point-min) (point-max))" \
           -f save-buffer

    if [ $? -eq 0 ]; then
        echo "✓ $file formatted successfully"
    else
        echo "✗ Error formatting $file"
        return 1
    fi
}

# 1. Determine which files to process
if [ $# -gt 0 ]; then
    # Use files passed as arguments
    FILES=("$@")
else
    # Find all *.lisp files recursively in the current directory
    echo "No arguments provided. Searching for all *.lisp files recursively..."
    mapfile -t FILES < <(find . -type f -name "*.lisp")
fi

# Check if any files were found/provided
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .lisp files found to format."
    exit 0
fi

# 2. Iterate through the files and format them
EXIT_CODE=0
for file in "${FILES[@]}"; do
    if ! format_lisp_file "$file"; then
        EXIT_CODE=1
    fi
done

# Exit with 1 if any file failed to format
exit $EXIT_CODE
