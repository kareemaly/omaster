#! /bin/bash

radon cc -s -a -n C ./omaster
jscpd ./omaster
vulture ./omaster