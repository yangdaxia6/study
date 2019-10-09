#!/bin/bash

alig=true

if [ "$alig" = true ]; then
  echo 'jiaozheng'
  python test.py $alig 
else
  echo 'no!'
  python test.py $alig
fi
