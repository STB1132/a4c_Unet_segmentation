#!/bin/bash
for f in *.png           # no need to use ls.
do
    filename=${f##*/}          # Use the last part of a path.
    echo filename

    filename=${filename%.*}    # Remove from the last dot.
    dir=$filename/masks       # Remove "tv" in front of filename.

    echo "$filename $dir"
    if [[ -d $dir ]]; then     # If the directory exists
        mv "$f" "$dir"/ # Move file there.
    fi
done
