for file in *.avi; do ffmpeg -i "$file" "${file%.avi}".jpeg; done
