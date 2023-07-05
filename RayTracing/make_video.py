import cv2 

n_frames = int(input("Enter number of frames:\n"))
frames = [cv2.imread("frames/out" + str(i) + ".ppm") for i in range(n_frames)]

video = cv2.VideoWriter("frames/my_video.avi", 0, n_frames // 10, (len(frames[0][0]), len(frames[0]))) 
# Добавление изображений к видео по одному
for frame in frames: 
    video.write(frame) 

video.release()  # выпуск сгенерированного видео
