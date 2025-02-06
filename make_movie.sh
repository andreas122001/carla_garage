module load FFmpeg/6.0-GCCcore-12.3.0

ffmpeg -framerate 20 -i database/dataset_test1/data/data/routes_validation_routeroutes_validation_route0_10_11_18_18_48_10_11_18_18_48/rgb_augmented/%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
