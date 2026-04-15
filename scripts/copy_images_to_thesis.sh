#! /bin/bash

OUTPUTS="./outputs"
DEST="./thesis/images"
DIAGNOSTICS="$OUTPUTS/diagnostics"
FRAMES="$OUTPUTS/temporal_maps/animation_full_inference/frames"

echo $DIAGNOSTICS
echo $FRAMES

cp $DIAGNOSTICS/*.png $DEST

cp $FRAMES/frame_001.png $DEST
cp $FRAMES/frame_028.png $DEST
cp $FRAMES/frame_034.png $DEST
cp $FRAMES/frame_066.png $DEST
