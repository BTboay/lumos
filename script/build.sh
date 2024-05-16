#!/bin/bash

VERSION=v0.1
CDIR=`pwd`

BUILDDIR=$CDIR/build

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

mkdir $BUILDDIR
wait

cp -r $CDIR/include $BUILDDIR/include
cp -r $CDIR/lumos $BUILDDIR/lumos
cp -r $CDIR/lib $BUILDDIR/lib
cp $CDIR/script/makefile $BUILDDIR/makefile

tar -zcvf lumos.tar.gz ./build
cat $CDIR/script/install.sh lumos.tar.gz > lumos-$VERSION.run

rm -f lumos.tar.gz

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi