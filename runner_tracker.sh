#!/usr/bin/env python

count=0
st_point=0
while read net; do
  if [ $count -ge $st_point ]
    then
      echo $net
      python trainer_tester.py "$net" "$count"
  fi
  count=$(($count+1))
  echo $count
done <nets_list20_7.txt
