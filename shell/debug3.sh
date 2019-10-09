#!/bin/bash

alig=true

datas=(ccpd blue yellow)
for data in ${datas[@]}
do
  if [ "$alig" = true ]; then
    echo 'jiaozheng'
    python test.py $alig /mnt/$data/${data}".txt"
  else
    echo 'no!'
    python test.py $alig /mnt/$data.txt
  fi
done
