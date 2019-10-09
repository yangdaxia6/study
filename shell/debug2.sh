#!/bin/bash

alig=true

data=ccpd

if [ "$alig" = true ]; then
  echo 'jiaozheng'
  python test.py $alig /mnt/$data/${data}".txt"
else
  echo 'no!'
  python test.py $alig /mnt/$data.txt
fi
