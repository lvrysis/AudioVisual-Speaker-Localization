import csv as csv
# import Picture as picture
import Video as video

# print "Python Version" + sys.version
# print "Numpy version " + np.__version__
# print "Video Version " + cv.__version__


def write_csv(path, data):
    csv_file = open(path, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
    csv_writer.writerows(data)
    # csv_writer.writerow(data[i])


# frames, analysis = video.capture(320, 240, 30)
frames, analysis = video.read('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/00001.mp4', 25)

# video.save('Output/output.avi', frames, 25)
# write_csv('Output/output.csv', analysis)