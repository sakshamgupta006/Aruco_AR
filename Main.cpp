///*****************************
//Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

//Redistribution and use in source and binary forms, with or without modification, are
//permitted provided that the following conditions are met:

//   1. Redistributions of source code must retain the above copyright notice, this list of
//      conditions and the following disclaimer.

//   2. Redistributions in binary form must reproduce the above copyright notice, this list
//      of conditions and the following disclaimer in the documentation and/or other materials
//      provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
//WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
//FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
//CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
//ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//The views and conclusions contained in the software and documentation are those of the
//authors and should not be interpreted as representing official policies, either expressed
//or implied, of Rafael Muñoz Salinas.
//********************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "aruco/aruco.h"
#include "aruco/posetracker.h"
#include "aruco/cvdrawingutils.h"
using namespace cv;
using namespace aruco;
///************************************
// *
// *
// *
// *
// ************************************/

////input.avi board_config.yml camera.yml  0.033
//// P8.mov board_new_config.yml studio_output.yml  0.5
///

void ButterworthLowpassFilter0100SixthOrder(const double src[], double dest[], int size)
{
    const int NZEROS = 6;
    const int NPOLES = 6;
    const double GAIN = 2.936532839e+03;

    double xv[NZEROS+1] = {0.0}, yv[NPOLES+1] = {0.0};

    for (int i = 0; i < size; i++)
    {
        xv[0] = xv[1]; xv[1] = xv[2]; xv[2] = xv[3]; xv[3] = xv[4]; xv[4] = xv[5]; xv[5] = xv[6];
        xv[6] = src[i] / GAIN;
        yv[0] = yv[1]; yv[1] = yv[2]; yv[2] = yv[3]; yv[3] = yv[4]; yv[4] = yv[5]; yv[5] = yv[6];
        yv[6] =   (xv[0] + xv[6]) + 6.0 * (xv[1] + xv[5]) + 15.0 * (xv[2] + xv[4])
                     + 20.0 * xv[3]
                     + ( -0.0837564796 * yv[0]) + (  0.7052741145 * yv[1])
                     + ( -2.5294949058 * yv[2]) + (  4.9654152288 * yv[3])
                     + ( -5.6586671659 * yv[4]) + (  3.5794347983 * yv[5]);
        dest[i] = yv[6];
    }
}
// Calculates rotation matrix given euler angles.
Mat euler2rot(Vec3f theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);


    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;

    return R;

}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());

    return  norm(I, shouldBeIdentity) < 1e-6;

}

Vec3f rot2euler(Mat R)
{

    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);



}

void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
  KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
  cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));       // set process noise
  cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-3));   // set measurement noise
  cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
                 /* DYNAMIC MODEL */
  //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
  // position
  KF.transitionMatrix.at<double>(0,3) = dt;
  KF.transitionMatrix.at<double>(1,4) = dt;
  KF.transitionMatrix.at<double>(2,5) = dt;
  KF.transitionMatrix.at<double>(3,6) = dt;
  KF.transitionMatrix.at<double>(4,7) = dt;
  KF.transitionMatrix.at<double>(5,8) = dt;
  KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
  // orientation
  KF.transitionMatrix.at<double>(9,12) = dt;
  KF.transitionMatrix.at<double>(10,13) = dt;
  KF.transitionMatrix.at<double>(11,14) = dt;
  KF.transitionMatrix.at<double>(12,15) = dt;
  KF.transitionMatrix.at<double>(13,16) = dt;
  KF.transitionMatrix.at<double>(14,17) = dt;
  KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
       /* MEASUREMENT MODEL */
  //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
  KF.measurementMatrix.at<double>(0,0) = 1;  // x
  KF.measurementMatrix.at<double>(1,1) = 1;  // y
  KF.measurementMatrix.at<double>(2,2) = 1;  // z
  KF.measurementMatrix.at<double>(3,9) = 1;  // roll
  KF.measurementMatrix.at<double>(4,10) = 1; // pitch
  KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

void fillMeasurements( cv::Mat &measurements,
                   const cv::Mat &translation_measured, const cv::Mat &rotation_measured)
{
    // Convert rotation matrix to euler angles
    cv::Mat measured_eulers(3, 1, CV_64F);
    Vec3f euler= rot2euler(rotation_measured);
    measured_eulers.at<double>(0)=euler[0];
    measured_eulers.at<double>(1)=euler[1];
    measured_eulers.at<double>(2)=euler[2];

    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                     cv::Mat &translation_estimated, cv::Mat &rotation_estimated )
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

Mat draw(Mat img, vector<Point2f> imgpts)
{
    vector<Point> first,last;
    vector<vector<Point>> top,bottom,up,down,left,right;
    for(int i=0;i<imgpts.size();i++)
    {
        if(i>3)
            last.push_back(imgpts[i]);
        else
            first.push_back(imgpts[i]);
    }
     Mat cnt=Mat::zeros(cvSize(img.cols,img.rows),CV_8UC1);
//    for(int i=0;i<4;i++)
//    {
//        line(img,first[i],last[i],(255),3);

//        int t= (i+1<4)?i+1:0;
//        line(img,first[i],first[t],(255),3);

//        line(img,last[i],last[t],(255),3);

//    }
////circle beg
//    Point center1,center2,center;

//    center1.x=(first[0].x + first[2].x)/2;
//    center1.y=(first[0].y + first[2].y)/2;

//    center2.x=(last[0].x + last[2].x)/2;
//    center2.y=(last[0].y + last[2].y)/2;

//    center.x=(center1.x + center2.x)/2;
//    center.y=(center1.y + center2.y)/2;


//    //double radius= sqrt(pow(first[0].x - last[2].x,2) + pow(first[0].y - last[2].y,2))/2;
//    double radius = (abs(first[0].y -first[1].y))/2;
////    cout<<center<<endl;
//    circle(img,center,radius,cvScalar(0,0,255),-1);


//    for(int i=0;i<4;i++)
//    {
//        line(img,first[i],last[i],(255),3);

//        int t= (i+1<4)?i+1:0;
//        line(img,first[i],first[t],(255),3);

//        line(img,last[i],last[t],(255),3);

//    }
////circle end
    top.push_back(first);
    bottom.push_back(last);
///pyramid begins
vector<Point> F1,F2,F3,F4;
Point vertex;
vertex.x= (first[0].x+first[2].x)/2;
vertex.y= (first[0].y+first[2].y)/2;

F1.push_back(last[0]);
F1.push_back(last[1]);
F1.push_back(vertex);
up.push_back(F1);

F2.push_back(last[1]);
F2.push_back(last[2]);
F2.push_back(vertex);
right.push_back(F2);

F3.push_back(last[2]);
F3.push_back(last[3]);
F3.push_back(vertex);
down.push_back(F3);

F4.push_back(last[3]);
F4.push_back(last[0]);
F4.push_back(vertex);
left.push_back(F4);

drawContours(img,bottom,-1,cvScalar(0,255,0),CV_FILLED);
drawContours(img,down,-1,cvScalar(0,0,255),CV_FILLED);
drawContours(img,right,-1,cvScalar(255,0,0),CV_FILLED);
drawContours(img,left,-1,cvScalar(255,255,255),CV_FILLED);
drawContours(img,up,-1,cvScalar(125,125,0),CV_FILLED);

line(img,last[0],vertex,(255),3);
line(img,last[1],vertex,(255),3);
line(img,last[2],vertex,(255),3);
line(img,last[3],vertex,(255),3);

//for(int i=0;i<4;i++)
//{
//    line(img,first[i],last[i],(255),3);

//    int t= (i+1<4)?i+1:0;
//    line(img,first[i],first[t],(255),3);

//    line(img,last[i],last[t],(255),3);

//}
/// pyramid ends
    vector<Point> U,D,L,R,P;
    //up face points
    U.push_back(first[0]);
    U.push_back(first[1]);
    U.push_back(last[1]);
    U.push_back(last[0]);
    //up.push_back(U);

    //down face points
    D.push_back(first[2]);
    D.push_back(first[3]);
    D.push_back(last[3]);
    D.push_back(last[2]);
    //down.push_back(D);

    //left face points
    L.push_back(first[0]);
    L.push_back(first[3]);
    L.push_back(last[3]);
    L.push_back(last[0]);
    //left.push_back(L);

    //right face points
    R.push_back(first[1]);
    R.push_back(first[2]);
    R.push_back(last[2]);
    R.push_back(last[1]);
    //right.push_back(R);


//    drawContours(img,top,-1,cvScalar(0,0,255),CV_FILLED);
//    drawContours(img,up,-1,cvScalar(255,0,0),CV_FILLED);
//    drawContours(img,down,-1,cvScalar(125,125,0),CV_FILLED);
//    drawContours(img,left,-1,cvScalar(125,0,125),CV_FILLED);
//    drawContours(img,right,-1,cvScalar(0,125,125),CV_FILLED);
//    drawContours(img,bottom,-1,cvScalar(0,255,0),CV_FILLED);

    //drawContours(img,top,-1,cvScalar(0,255,0));
    return img;
}


int main(int argc, char **argv) {

    double signal[6][10000];
    double out_signal[6][10000];
    Mat raw[10000];
    int iter=0;
    int marker_size=0.35;
    aruco::CameraParameters param_cam;
    try {
        if (argc < 3) {
            cerr << "Usage: (in_image|video.avi)  markerSetConfig.yml [cameraParams.yml] [markerSize]  [outImage]" << endl;
            exit(0);
        }
        //open video
        cv::Mat InImage;
       VideoCapture vreader(argv[1]);
//        cout<<"enter device id"<<endl;
//        int id;
//        cin>>id;
//        VideoCapture vreader(id);
        VideoWriter  vcap, markers,smooth_pyr;
         vcap.open("./aruco_pose.avi",CV_FOURCC('M','J','P','G'),20.0,cvSize((int) vreader.get(CV_CAP_PROP_FRAME_WIDTH),(int) vreader.get(CV_CAP_PROP_FRAME_HEIGHT)));
         markers.open("./aruco_markers.avi",CV_FOURCC('M','J','P','G'),20.0,cvSize((int) vreader.get(CV_CAP_PROP_FRAME_WIDTH),(int) vreader.get(CV_CAP_PROP_FRAME_HEIGHT)));
         smooth_pyr.open("./pyramid_butterworth.avi",CV_FOURCC('M','J','P','G'),20.0,cvSize((int) vreader.get(CV_CAP_PROP_FRAME_WIDTH),(int) vreader.get(CV_CAP_PROP_FRAME_HEIGHT)));
         //initate kalman filter
         cv::KalmanFilter KF;         // instantiate Kalman Filter
         int nStates = 18;            // the number of states
         int nMeasurements = 6;       // the number of measured states
         int nInputs = 0;             // the number of action control
         double dt = 0.05;           // time between measurements (1/FPS)
         int minInliersKalman= 12;
         initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function


        while(1==1)
        {
            // if(iter>900)
            // {
            //     cout<<"rendering smoothed frames"<<endl;
            //     break;
            // }

            if (!vreader.isOpened()) throw std::runtime_error("Could not open input");
            //read input image(or first image from video)
            vreader>>InImage;

            //thresh start
            //erode(img_bw, img_final, Mat(), Point(-1, -1), 2, 1, 1);
//            Mat gray;



//            cv::Mat input,lab_image,tmp_plane;
//               InImage.copyTo(input);
//               cv::cvtColor(input, lab_image, CV_BGR2Lab);

//               // Extract the L channel
//               std::vector<cv::Mat> lab_planes(3);
//               cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
//               //imshow("plane1",lab_planes[0]);
//               //imshow("plane2",lab_planes[1]);
//               //imshow("plane3",lab_planes[2]);

//               lab_planes[0].copyTo(tmp_plane);
//               // apply the CLAHE algorithm to the L channel
//               cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
//               clahe->setClipLimit(4);
//               cv::Mat dst;
//               clahe->apply(lab_planes[0], dst);

//               // Merge the the color planes back into an Lab image
//               dst.copyTo(lab_planes[0]);
//               cv::merge(lab_planes, lab_image);

//              // convert back to RGB
//              cv::Mat image_clahe;
//              cv::cvtColor(lab_image, input, CV_Lab2BGR);

//               input.copyTo(gray);


////            cvtColor( gray, gray, CV_BGR2GRAY );
////            //equalizeHist(gray,gray);
////            medianBlur(gray,gray,5);
//////            normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
////            //GaussianBlur(gray,gray,cvSize(3,3),0,0);

//// //           adaptiveThreshold(gray,gray,150,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2);
////            threshold( gray, gray, 105, 255,0);

////            //GaussianBlur(tmp_plane,tmp_plane,cvSize(3,3),0,0);
////           //adaptiveThreshold(tmp_plane,tmp_plane,60,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2);
////            normalize(tmp_plane, tmp_plane, 0, 255, NORM_MINMAX, CV_8UC1);
////            //equalizeHist(tmp_plane,tmp_plane);
////           threshold( tmp_plane, tmp_plane,160, 255,0);
////            imshow("thresh",gray);
////            //thresh end

////adaptive filter start
//            Mat channels[3];
//            split(input,channels);



//            adaptiveThreshold(channels[1],gray,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,31,11);
//            //erosion
//            int erosion_type=MORPH_CROSS,erosion_size=3;
////              if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
////              else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
////              else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

//              Mat element = getStructuringElement( erosion_type,
//                                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                                                   Point( erosion_size, erosion_size ) );

//              /// Apply the erosion operation
//              Mat mask;
//              gray.copyTo(mask);
//              //dilate( mask, mask, element );
//              erode(mask,mask,element);


//            //

//            imshow("input image",gray);
////adaptive filter end
            //read marker map
            MarkerMap TheMarkerMapConfig;//configuration of the map
            TheMarkerMapConfig.readFromFile(argv[2]);

            //read camera params if indicated
            aruco::CameraParameters CamParam;
            if (argc >= 4)
            {
                CamParam.readFromXMLFile(argv[3]);
                // resizes the parameters to fit the size of the input image
                CamParam.resize(InImage.size());
            }

            param_cam=CamParam;
        //read marker size if indicated
            float MarkerSize = -1;
            if (argc >= 5) MarkerSize = atof(argv[4]);
            //transform the markersetconfig to meter if is in pixels and the markersize indicated
            if ( TheMarkerMapConfig.isExpressedInPixels() && MarkerSize>0)
                TheMarkerMapConfig=TheMarkerMapConfig.convertToMeters(MarkerSize);

              cout<<TheMarkerMapConfig[0]<<"Markermap"<<endl;

         //Let go
            MarkerDetector MDetector;
            //set the appropiate dictionary type so that the detector knows it
            MDetector.setDictionary(TheMarkerMapConfig.getDictionary());
            // detect markers without computing R and T information
            vector< Marker >  Markers=MDetector.detect(InImage);//(InImage);

            //print the markers detected that belongs to the markerset
            vector<int> markers_from_set=TheMarkerMapConfig.getIndices(Markers);

            Mat tmp;
            InImage.copyTo(tmp);
            for(auto idx:markers_from_set) Markers[idx].draw(tmp, Scalar(0, 0, 255), 2);
            markers<<tmp;
            imshow("detected_markers",tmp);

        cout<<TheMarkerMapConfig.isExpressedInMeters()<<" "<<CamParam.isValid() <<endl;
            //detect the 3d camera location wrt the markerset (if possible)
            if ( TheMarkerMapConfig.isExpressedInMeters() && CamParam.isValid())
            {

                MarkerMapPoseTracker MSPoseTracker;//tracks the pose of the marker map
                MSPoseTracker.setParams(CamParam,TheMarkerMapConfig);
                if ( MSPoseTracker.estimatePose(Markers) )//if pose correctly computed, print the reference system
                {
                    // Get the measured translation
                     cv::Mat measurements(6, 1, CV_64F);
                    cv::Mat translation_measured(3, 1, CV_64F);
                    //translation_measured = MSPoseTracker.getTvec();
                    // Get the measured rotation
                    cv::Mat RT(4,4,CV_64FC1),rotation_measured(3, 3, CV_64F);
                    //rotation_measured = MSPoseTracker.getRvec();
                    RT=MSPoseTracker.getRTMatrix();
                    cout<<RT<<endl;

                    for(int row=0;row<3;row++)
                    {
                        for(int col=0;col<3;col++)
                           rotation_measured.at<double>(row,col)=RT.at<float>(row,col);
                    }
                    translation_measured.at<double>(0)=RT.at<float>(0,3);
                    translation_measured.at<double>(1)=RT.at<float>(1,3);
                    translation_measured.at<double>(2)=RT.at<float>(2,3);
                    //cout<<rotation_measured<<endl<<RT;
                    // fill the measurements vector

                    // Instantiate estimated translation and rotation
                    cv::Mat translation_estimated(3, 1, CV_64F);
                    cv::Mat rotation_estimated(3, 3, CV_64F);
                    fillMeasurements(measurements, translation_measured, rotation_measured);

                    if(Markers.capacity()>=minInliersKalman && false)
                    {

                            // update the Kalman filter with good measurements
                            updateKalmanFilter( KF, measurements,
                                          translation_estimated, rotation_estimated);



                            vector<Point2f> imagePoints;
                            vector<Point3f> axis;
                            axis.push_back(Point3f(-0.4,0.6,2.2));
                            axis.push_back(Point3f(-0.6,0.6,2.2));
                            axis.push_back(Point3f(-0.6,0.6,2));
                            axis.push_back(Point3f(-0.4,0.6,2));
                            axis.push_back(Point3f(-0.4,0.8,2.2));
                            axis.push_back(Point3f(-0.6,0.8,2.2));
                            axis.push_back(Point3f(-0.6,0.8,2));
                            axis.push_back(Point3f(-0.4,0.8,2));
                            projectPoints(axis, rotation_estimated, translation_estimated, CamParam.CameraMatrix, CamParam.Distorsion,imagePoints);
                            Mat in_copy;
                            InImage.copyTo(in_copy);
                            Mat res=draw(in_copy,imagePoints);
                            imshow("virtual pyramid",res);
                            vcap<<res;
                           cout<<"rotation_val"<<endl<<rotation_estimated<<endl<<rotation_measured<<endl;
                           cout<<"translation_val"<<endl<<translation_estimated<<endl<<translation_measured<<endl;

                           aruco::CvDrawingUtils::draw3dAxis(InImage,CamParam,rotation_estimated,translation_estimated,TheMarkerMapConfig[0].getMarkerSize()*2);

                    }
                    else
                    {
                        // InImage.copyTo(raw[iter]);

                        vector<Point2f> imagePoints,imagePoints2,imagePoints3;
                        vector<Point3f> axis,axis2,axis3;
//                        axis.push_back(Point3f(-3,0,22));
//                        axis.push_back(Point3f(-5,0,22));
//                        axis.push_back(Point3f(-5,0,20));
//                        axis.push_back(Point3f(-3,0,20));
//                        axis.push_back(Point3f(-3,-2,22));
//                        axis.push_back(Point3f(-5,-2,22));
//                        axis.push_back(Point3f(-5,-2,20));
//                        axis.push_back(Point3f(-3,-2,20));
                          //on one ground marker
                        axis.push_back(Point3f(1.058,0.66,0.337));
                        axis.push_back(Point3f(1.395,0.66,0.337));
                        axis.push_back(Point3f(1.395,0.997,0.337));
                        axis.push_back(Point3f(1.058,0.997,0.337));
                        axis.push_back(Point3f(1.058,0.66,0));
                        axis.push_back(Point3f(1.395,0.66,0));
                        axis.push_back(Point3f(1.395,0.997,0));
                        axis.push_back(Point3f(1.058,0.997,0));

                        axis2.push_back(Point3f(1.058,0.33,0.337));
                        axis2.push_back(Point3f(1.395,0.33,0.337));
                        axis2.push_back(Point3f(1.395,0.667,0.337));
                        axis2.push_back(Point3f(1.058,0.667,0.337));
                        axis2.push_back(Point3f(1.058,0.33,0));
                        axis2.push_back(Point3f(1.395,0.33,0));
                        axis2.push_back(Point3f(1.395,0.667,0));
                        axis2.push_back(Point3f(1.058,0.667,0));

                        axis3.push_back(Point3f(1.058,0.0,0.337));
                        axis3.push_back(Point3f(1.395,0.0,0.337));
                        axis3.push_back(Point3f(1.395,0.337,0.337));
                        axis3.push_back(Point3f(1.058,0.337,0.337));
                        axis3.push_back(Point3f(1.058,0.0,0));
                        axis3.push_back(Point3f(1.395,0.0,0));
                        axis3.push_back(Point3f(1.395,0.337,0));
                        axis3.push_back(Point3f(1.058,0.337,0));

                        // wall plane parellel
                        // axis.push_back(Point3f(1.058,0.337,0.66));
                        // axis.push_back(Point3f(1.395,0.337,0.66));
                        // axis.push_back(Point3f(1.395,0.337,0.997));
                        // axis.push_back(Point3f(1.058,0.337,0.997));
                        // axis.push_back(Point3f(1.058,0,0.66));
                        // axis.push_back(Point3f(1.395,0,0.66));
                        // axis.push_back(Point3f(1.395,0,0.997));
                        // axis.push_back(Point3f(1.058,0,0.997));

                        // axis.push_back(Point3f(1.058,0.66,0.674));
                        // axis.push_back(Point3f(1.395,0.66,0.674));
                        // axis.push_back(Point3f(1.395,0.997,0.674));
                        // axis.push_back(Point3f(1.058,0.997,0.674));
                        // axis.push_back(Point3f(1.058,0.66,0.337));
                        // axis.push_back(Point3f(1.395,0.66,0.337));
                        // axis.push_back(Point3f(1.395,0.997,0.337));
                        // axis.push_back(Point3f(1.058,0.997,0.337));

                        projectPoints(axis, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints);
                         projectPoints(axis2, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints2);
                          projectPoints(axis3, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints3);
                        Mat in_copy;
                        InImage.copyTo(in_copy);
                        draw(in_copy,imagePoints);
                        draw(in_copy,imagePoints2);
                        Mat res=draw(in_copy,imagePoints3);
                        vcap<<res;
                        imshow("virtual pyramid",res);
                       // cout<<MSPoseTracker.getRTMatrix();
                        updateKalmanFilter( KF, measurements,
                                      translation_estimated, rotation_estimated);
                        cout<<"ROT:"<<MSPoseTracker.getRvec()<<endl;
                        aruco::CvDrawingUtils::draw3dAxis(InImage,CamParam,MSPoseTracker.getRvec(),MSPoseTracker.getTvec(),TheMarkerMapConfig[0].getMarkerSize()*2);
                        marker_size=TheMarkerMapConfig[0].getMarkerSize()*2;
                        // signal[0][iter]= MSPoseTracker.getRvec().at<float>(0);
                        // signal[1][iter]= MSPoseTracker.getRvec().at<float>(1);
                        // signal[2][iter]= MSPoseTracker.getRvec().at<float>(2);
                        // signal[3][iter]= MSPoseTracker.getTvec().at<float>(0);
                        // signal[4][iter]= MSPoseTracker.getTvec().at<float>(1);
                        // signal[5][iter]= MSPoseTracker.getTvec().at<float>(2);

                        // iter++;
                    }

                }
                else
                {
                  vcap<<InImage;


                }

            }
        // show input with augmented information
            cv::imshow("in", InImage);
            //vcap<<InImage;
            waitKey(20);
        //while ( char(cv::waitKey(0))!=27) ; // wait for esc to be pressed
        //save output if indicated
        if (argc >= 6) cv::imwrite(argv[5], InImage);
        }

//         ButterworthLowpassFilter0100SixthOrder(signal[0],out_signal[0],iter);
//         ButterworthLowpassFilter0100SixthOrder(signal[1],out_signal[1],iter);
//         ButterworthLowpassFilter0100SixthOrder(signal[2],out_signal[2],iter);
//         ButterworthLowpassFilter0100SixthOrder(signal[3],out_signal[3],iter);
//         ButterworthLowpassFilter0100SixthOrder(signal[4],out_signal[4],iter);
//         ButterworthLowpassFilter0100SixthOrder(signal[5],out_signal[5],iter);
//         for(int i=0;i<iter;i++)
//         {
//             cv::Mat rotation_smoothed(3, 1, CV_64F);
//             cv::Mat translation_smoothed(3, 1, CV_64F);
//             rotation_smoothed.at<double>(0)=out_signal[0][i];
//             rotation_smoothed.at<double>(1)=out_signal[1][i];
//             rotation_smoothed.at<double>(2)=out_signal[2][i];
//             translation_smoothed.at<double>(0)=out_signal[3][i];
//             translation_smoothed.at<double>(1)=out_signal[4][i];
//             translation_smoothed.at<double>(2)=out_signal[5][i];

//             vector<Point2f> imagePoints;
//             vector<Point3f> axis;
// //            axis.push_back(Point3f(-3,0,22));
// //            axis.push_back(Point3f(-5,0,22));
// //            axis.push_back(Point3f(-5,0,20));
// //            axis.push_back(Point3f(-3,0,20));
// //            axis.push_back(Point3f(-3,-2,22));
// //            axis.push_back(Point3f(-5,-2,22));
// //            axis.push_back(Point3f(-5,-2,20));
// //            axis.push_back(Point3f(-3,-2,20));
//             axis.push_back(Point3f(-0.4,0.6,2.2));
//             axis.push_back(Point3f(-0.6,0.6,2.2));
//             axis.push_back(Point3f(-0.6,0.6,2));
//             axis.push_back(Point3f(-0.4,0.6,2));
//             axis.push_back(Point3f(-0.4,0.8,2.2));
//             axis.push_back(Point3f(-0.6,0.8,2.2));
//             axis.push_back(Point3f(-0.6,0.8,2));
//             axis.push_back(Point3f(-0.4,0.8,2));
//             projectPoints(axis,rotation_smoothed,translation_smoothed,param_cam.CameraMatrix, param_cam.Distorsion,imagePoints);
//             Mat res=draw(raw[i],imagePoints);
//             imshow("smoothed",res);
//             smooth_pyr<<res;
//             waitKey(20);

//         }


    } catch (std::exception &ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
}