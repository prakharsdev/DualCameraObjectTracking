#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    // Load the videos from the two cameras
    VideoCapture cap1(0);
    VideoCapture cap2(1);

    // Extract keypoints from the first frame of each video
    Mat frame1, frame2, hsv_frame1, hsv_frame2;
    cap1 >> frame1;
    cap2 >> frame2;

    cvtColor(frame1, hsv_frame1, COLOR_BGR2HSV);
    cvtColor(frame2, hsv_frame2, COLOR_BGR2HSV);

    // Define the range of red color
    Scalar lower_red = Scalar(0, 70, 50);
    Scalar upper_red = Scalar(10, 255, 255);

    // threshold the hsv image to get only red colors
    Mat mask1, mask2;
    inRange(hsv_frame1, lower_red, upper_red, mask1);
    inRange(hsv_frame2, lower_red, upper_red, mask2);
    
    // Extract keypoints from the mask
    vector<KeyPoint> kp1, kp2;
    Ptr<ORB> detector = ORB::create();
    detector->detect(mask1, kp1);
    detector->detect(mask2, kp2);

    // Track the keypoints across the frames
    vector<Point2f> prev_kp1, prev_kp2;
    vector<Point2f> next_kp1, next_kp2;
    vector<uchar> status1, status2;
    vector<float> error1, error2;
    Mat gray1, gray2;

    while (true) {
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty())
            break;

        cvtColor(frame1, gray1, COLOR_BGR2GRAY);
        cvtColor(frame2, gray2, COLOR_BGR2GRAY);

        prev_kp1 = next_kp1;
        prev_kp2 = next_kp2;

        calcOpticalFlowPyrLK(gray1, gray2, prev_kp1, next_kp1, status1, error1);
        calcOpticalFlowPyrLK(gray2, gray1, prev_kp2, next_kp2, status2, error2);

        // Align the frames in time
        double time_offset = find_best_offset(next_kp1, next_kp2);

        // Draw rectangle on the object in both frames
        Rect roi1 = find_object_roi(next_kp1, time_offset);
        Rect roi2 = find_object_roi(next_kp2, time_offset);
        rectangle(frame1, roi1, Scalar(0, 255, 0), 2);
        rectangle(frame2, roi2, Scalar(0, 255, 0), 2);

