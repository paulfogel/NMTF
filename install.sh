#!/bin/bash
publish=false

PACKAGE="nmtf"

while true; do
  case "$1" in
    -p | --publish) publish=true ; shift 1 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

pip3 uninstall "$PACKAGE" -y || echo "No $PACKAGE to uninstall"
pip3 install setuptools
python3 setup.py install
if $publish && [ -f "$HOME/bin/gspip" ] ; then
  gspip push -s "  "
fi
if [ -d "dist" ] ; then rm -r dist ; fi
if [ -d "build" ] ; then rm -r build ; fi
if ls "$PACKAGE".egg-info* &> /dev/null ; then rm -r "$PACKAGE".egg-info* ; fi
