from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import utils

test_img = utils.loadImage('./test_images/test1.jpg')

plt.ion()
display = plt.imshow(test_img)

def test(img):
    print(img.shape)
    display.set_data(img)
    plt.pause(0.05)
    return img


path ='test_video.mp4'
video = VideoFileClip(path)
video_clip = video.fl_image(test)
video_clip.write_videofile('./videos/mess.mp4', audio=False)
