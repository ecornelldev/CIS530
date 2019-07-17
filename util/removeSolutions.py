#!/usr/bin/python

import sys,re

total=''
cutout=False
for l in sys.stdin:
	if '### BEGIN SOLUTION' in l: 
		cutout=True
	if '### END SOLUTION' in l:
		cutout=False
		l=l.replace('### END SOLUTION','### YOUR CODE HERE!')
	if cutout: continue
	total+=l


print(total)