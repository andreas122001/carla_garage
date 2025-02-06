module load FFmpeg/6.0-GCCcore-12.3.0

ffmpeg -framerate 20 -start_number 5 -i ./%04d.png -c:v libx264 -pix_fmt yuv420p ../$(date +"%y.%m.%d-%X").mp4