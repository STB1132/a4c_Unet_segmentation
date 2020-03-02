i=1
for i in *.avi; do
mkdir -p ${i%.avi}/{images,masks}
done
