from ctypes.wintypes import RGB
import cv2
import numpy as np


# rgb_img = np.load(f'./data/tina/dough.npy')



# #Displays image inside a window
# cv2.imshow('color image',rgb_img)

# # Waits for a keystroke
# cv2.waitKey(0)  



# rgb_img[:50, :50, :] = 0

# cv2.imshow('color image',rgb_img)

# cv2.waitKey(0)  

# Destroys all the windows created
# cv2.destroyAllWindows() 

def main():
    WINDOW_TITLE_PREFIX = 'Robotic Dough Shaping - '
    IMG_SHAPE = (480, 640)
    # Region of interest (in pixels)
    # Note: x and y is swapped in array representation as compared to cv2 image visualization representation!
    # Here x and y are in the cv2 image visualization representation and (0,0) is in the top left corner
    ROI = {
        'x_min': 170,
        'y_min': 0,
        'x_max': 540,
        'y_max': 320
    }
    # Target shape detection parameters
    # Only circles with diameters 4 inch or more will be detected
    MIN_TARGET_CIRCLE_RADIUS = 50
    MAX_TARGET_CIRCLE_RADIUS = 180
    # Current shape detection parameters
    MIN_COLOR_INTENSITY = 70
    MIN_CONTOUR_AREA = 1000
    # Fraction of dough height reached at the roll start point
    DOUGH_HEIGHT_CONTRACTION_RATIO = 0.5
    # In meters
    # Z_OFFSET = 0.61

    RGB_IMG = np.load(f'./data/janko/dough-colors/green1.npy')

    def get_ROI_img(img):
        return img[ROI['y_min']:ROI['y_max'], ROI['x_min']:ROI['x_max']]


    ROI_rgb_img = get_ROI_img(RGB_IMG)

    # Color filter
    color_mask = np.zeros((*ROI_rgb_img.shape[:2], 3)).astype('uint8')
    color_mask[ROI_rgb_img < MIN_COLOR_INTENSITY] = 255
    overall_color_mask = cv2.bitwise_or(cv2.bitwise_or(color_mask[:, :, 0], color_mask[:, :, 1]), color_mask[:, :, 2])
    
    # Detect contours
    contours, _ = cv2.findContours(overall_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise ValueError(f'No contours detected for the current shape!')

    # Take the largest contour
    current_shape_contour = sorted(contours, key=lambda c: cv2.contourArea(c))[-1]

    current_shape_area = cv2.contourArea(current_shape_contour)
    if current_shape_area < MIN_CONTOUR_AREA:
        print(f'Warning: the area of the current shape is {current_shape_area} which is less than {MIN_CONTOUR_AREA}')

    # Undo ROI transform
    current_shape_contour[:, 0] += np.array([ROI['x_min'], ROI['y_min']])


    debug_img = RGB_IMG.copy()
    # Draw the region of interest
    cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
    # Draw the current shape






    # 1.) load stripes from image
    stripes = cv2.imread('./stripes_pattern.jpg')
    stripes = cv2.resize(stripes, (RGB_IMG.shape[1], RGB_IMG.shape[0]))

    stripes[:, :, 0] = 255


    # 2.) Generate stripes
    # stripes[stripes] = 
    # print((stripes[:, :] == np.array([0, 0, 0])).any())
    stripes = np.zeros(RGB_IMG.shape, dtype="uint8")
    for i in range(0, RGB_IMG.shape[1], 7):
        cv2.line(stripes, (i, 0), (i, RGB_IMG.shape[0]), color=(255, 0, 0), thickness=2)

    # stripes = stripes[100:100+RGB_IMG.shape[0], :RGB_IMG.shape[1]]
    # print(stripes.shape, RGB_IMG.shape) 
    # cv2.imshow('color image', stripes)
    # cv2.waitKey(0)

    overlay_bg = RGB_IMG.copy()
    current_shape_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    cv2.fillPoly(current_shape_mask, [current_shape_contour], color=255)

    overlay_bg = cv2.bitwise_and(overlay_bg, overlay_bg, mask=cv2.bitwise_not(current_shape_mask))
    overlay_current = cv2.bitwise_and(stripes, stripes, mask=current_shape_mask)

    # cv2.imshow('color image', overlay_current + overlay_bg)
    # cv2.waitKey(0)

    alpha = 0.2
    cv2.addWeighted(overlay_current + overlay_bg, alpha, debug_img, 1 - alpha, 0, debug_img)

    # 3.) Draw only the contour => best
    cv2.polylines(debug_img, [current_shape_contour], isClosed=True, color=(255, 0, 0), thickness=1)

    iou = 0.4352
    # Draw text
    cv2.rectangle(debug_img, (5, 5), (160, 130), color=(255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(debug_img, 'Target shape', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    # cv2.putText(debug_img, 'Current shape', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.putText(debug_img, f'IoU = {iou:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('color image', debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__=='__main__':
    # change()
    main()

# try putting subsriber before Interbotix code
# remove other code
