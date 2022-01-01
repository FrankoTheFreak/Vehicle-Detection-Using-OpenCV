# importing the packages
import cv2                       
import numpy as np


# capture input video stream
cap = cv2.VideoCapture("media files/inputvideo1.mp4")


# initialize background subtractor algorithm
bg = cv2.createBackgroundSubtractorMOG2()


# initialize loop
while True:                                       

    # read the input video stream frame by frame
    ret, frame = cap.read()

    if ret:

        # Step 1 : Image Processing

        # resize the frame to a fixed resolution,
        image = cv2.resize(src=frame, dsize=None, fx=0.9,
                           fy=0.9, interpolation=cv2.INTER_AREA)
        # check for resized video by uncommenting the line below
        # cv2.imshow("RESIZED VIDEO", image)                                                   

        # converts a colored image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check for grayscaled video by uncommenting the line below
        # cv2.imshow("BGR2GRAY", gray)

        # apply the background subtractor to image
        fgmask = bg.apply(gray)
        # check for background subtracted video by uncommenting the line below
        # cv2.imshow("BACKGROUND SUBTRACTOR", fgmask)


        # Step 2 : Morphological Operations

        # it ia a slider/filter that is used for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # it removes small holes present inside the foreground object  (car)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # check after applying closing by uncommenting the line below
        # cv2.imshow("CLOSING", closing) 

        # it removes the noise present in background
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # check after applying opening by uncommenting the line below
        # cv2.imshow("OPENING", opening)

        # it is applied after erosion which removes the noise but infact also shrinks the object so, dilation is used to enlarge it
        dilation = cv2.dilate(opening, kernel)
        # check after applying dilation by uncommenting the line below
        # cv2.imshow("DILATION", dilation)

        # here agruments are as follows -> (src img, threshold value, maximum value for pixels, thresholding operation) , this removes shadows from video
        thres, bin_img = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
        # check after applying thresholding by uncommenting the line below
        # cv2.imshow("THRESHOLDING", bin_img)


        # Step 3 : Centroid & Boundary

        # this gives contours and thier heirarchy in current frame
        contours, heirarchy = cv2.findContours(
            bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 500                   # defining minimum area for contour
        maxarea = 50000                 # defining maximum area for contour

        # contours at x co-ordinate in current frame
        cxx = np.zeros(len(contours))
        # contours at y co-ordinate in current frame
        cyy = np.zeros(len(contours))

        # looping through each contour in current frame
        for i in range(len(contours)):

            # using hierarchy to only consider parent contours (discarding contours within contours)
            if heirarchy[0, i, 3] == -1:

                # area of the contour
                area = cv2.contourArea(contours[i])

                if minarea < area < maxarea:                  # if the area lies between min and max threshold we continue

                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    # x co-ordinate of centroid
                    cx = int(M['m10'] / M['m00'])
                    # y co-ordinate of centroid
                    cy = int(M['m01'] / M['m00'])

                    # co-ordinates for bounding rectangle -> x, y = top left corner & w, h = width and height
                    x, y, w, h = cv2.boundingRect(cnt)

                    # draws a rectangle/boundary
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    # prints text representing centroids values
                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10,
                                cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)

                    # draws marker reprsenting centroids
                    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8,
                                   thickness=3, line_type=cv2.LINE_8)          


        cv2.imshow("VEHICLE DETECTION", image)

        if cv2.waitKey(1) == 13:                      # press [ENTER] to stop
            break

cap.release()
cv2.destroyAllWindows()
