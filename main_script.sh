#!/bin/bash

<<COMMENT
    @file main_script.sh

    @brief File to manage the different executions of the evolutionary procedure

    This work has been funded by the Spanish Ministry of Science, Innovation, and Universities under grant
    PGC2018-098813-B-C31 and ERDF funds

    @author Juan Carlos Gómez López

    @date 30/04/2020

    @copyright Licensed under GNU GPL-3.0-or-later

    «Copyright 2020 EffiComp@ugr.es»
COMMENT

# GA Parameters
Executions=$(grep -oP "(?<=<Executions>).*(?=</Executions)" config.xml)
ProjectPath=$(grep -oP "(?<=<ProjectPath>).*(?=</ProjectPath)" config.xml)

export PYTHONPATH=$ProjectPath:$PYTHONPATH

if [ "$Executions" -lt 1 ]; then
  echo 'Number of executions have to be bigger than 0'
  exit
fi
#export OMP_NUM_THREADS=1
# Run execution
for e in $(seq 1 "$Executions"); do
  if python3 "$ProjectPath"/src/main.py "$e"; then
    echo 'Genetic algorithm executed successfully'
  else
    echo 'Genetic algorithm can not be executed'
    exit
  fi
done

