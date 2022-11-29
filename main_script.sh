#!/bin/bash

<<COMMENT
  This file is part of Vegetto.

  Vegetto is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  Vegetto is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  Vegetto. If not, see <http://www.gnu.org/licenses/>.

  This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
  de Ciencia, Innovación y Universidades"), and by the European Regional
  Development Fund (ERDF).
COMMENT

Executions=$(grep -oP "(?<=<Executions>).*(?=</Executions)" config.xml)
ProjectPath=$(grep -oP "(?<=<ProjectPath>).*(?=</ProjectPath)" config.xml)

export PYTHONPATH=$ProjectPath:$PYTHONPATH

if [ "$Executions" -lt 1 ]; then
  echo 'Number of executions have to be bigger than 0'
  exit
fi

# Execute the wrapper
for e in $(seq 1 "$Executions"); do
  if python3 "$ProjectPath"/src/main.py "$e"; then
    echo 'Genetic algorithm executed successfully'
  else
    echo 'Genetic algorithm can not be executed'
    exit
  fi
done

