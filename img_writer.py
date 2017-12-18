import os, csv

with open("/home/varun/Courses/CIS680/vision_and_learning/HW4/datasets/cufs_photos/train_sketches.csv",'w') as f:
	w=csv.writer(f)
	for path, dirs, files in os.walk("/home/varun/Courses/CIS680/vision_and_learning/HW4/datasets/cufs_photos/train/sketches"):
		for filename in files:
			w.writerow([filename])

