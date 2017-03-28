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
        int count=1;
        vector<Point3f> axis;
        axis.push_back(Point3f(-1.6850000619888306e-01, 1.6850000619888306e-01, 0.));
axis.push_back(Point3f(1.6850000619888306e-01, 1.6850000619888306e-01, 0.));
axis.push_back(Point3f(1.6850000619888306e-01, -1.6850000619888306e-01, 0.));
axis.push_back(Point3f(-1.6850000619888306e-01, -1.6850000619888306e-01, 0.));
axis.push_back(Point3f(2.3476725816726685e-01, 1.8477736413478851e-01, -9.6186893060803413e-03));
axis.push_back(Point3f(5.7175493240356445e-01, 1.8402238190174103e-01, -1.2400219216942787e-02));
axis.push_back(Point3f(5.7100510597229004e-01, -1.5297619998455048e-01, -1.1770534329116344e-02));
axis.push_back(Point3f(2.3401743173599243e-01, -1.5222121775150299e-01, -8.9890044182538986e-03));
axis.push_back(Point3f(6.4542400836944580e-01, 1.7742897570133209e-01, -1.0616639629006386e-02));
axis.push_back(Point3f(9.8240739107131958e-01, 1.7902807891368866e-01, -1.3559970073401928e-02));
axis.push_back(Point3f(9.8401892185211182e-01, -1.5796510875225067e-01, -1.2142121791839600e-02));
axis.push_back(Point3f(6.4703553915023804e-01, -1.5956421196460724e-01, -9.1987913474440575e-03));
axis.push_back(Point3f(1.0566861629486084e+00, 1.7013075947761536e-01, -1.0338074527680874e-02));
axis.push_back(Point3f(1.3936718702316284e+00, 1.7149433493614197e-01, -1.3129827566444874e-02));
axis.push_back(Point3f(1.3950493335723877e+00, -1.6549876332283020e-01, -1.1470464058220387e-02));
axis.push_back(Point3f(1.0580636262893677e+00, -1.6686233878135681e-01, -8.6787110194563866e-03));
axis.push_back(Point3f(1.4652764797210693e+00, 1.7087981104850769e-01, -1.2704287655651569e-02));
axis.push_back(Point3f(1.8022723197937012e+00, 1.7229512333869934e-01, -1.3616259209811687e-02));
axis.push_back(Point3f(1.8036918640136719e+00, -1.6469827294349670e-01, -1.2052596546709538e-02));
axis.push_back(Point3f(1.4666960239410400e+00, -1.6611358523368835e-01, -1.1140624992549419e-02));
axis.push_back(Point3f(1.8741279840469360e+00, 1.7666725814342499e-01, -1.5781648457050323e-02));
axis.push_back(Point3f(2.2111217975616455e+00, 1.7798088490962982e-01, -1.7360242083668709e-02));
axis.push_back(Point3f(2.2124419212341309e+00, -1.5901373326778412e-01, -1.5988096594810486e-02));
axis.push_back(Point3f(1.8754479885101318e+00, -1.6032736003398895e-01, -1.4409502036869526e-02));
axis.push_back(Point3f(2.2831621170043945e+00, 1.8138086795806885e-01, -1.9563306123018265e-02));
axis.push_back(Point3f(2.6201546192169189e+00, 1.8259197473526001e-01, -2.1458117291331291e-02));
axis.push_back(Point3f(2.6213750839233398e+00, -1.5440168976783752e-01, -1.9790135324001312e-02));
axis.push_back(Point3f(2.2843825817108154e+00, -1.5561279654502869e-01, -1.7895324155688286e-02));
axis.push_back(Point3f(2.6933104991912842e+00, 1.8912550806999207e-01, -2.4505740031599998e-02));
axis.push_back(Point3f(3.0302910804748535e+00, 1.9073200225830078e-01, -2.7742223814129829e-02));
axis.push_back(Point3f(3.0319159030914307e+00, -1.4625871181488037e-01, -2.5835910812020302e-02));
axis.push_back(Point3f(2.6949353218078613e+00, -1.4786520600318909e-01, -2.2599427029490471e-02));
axis.push_back(Point3f(3.1040627956390381e+00, 1.9638308882713318e-01, -3.0864790081977844e-02));
axis.push_back(Point3f(3.4410324096679688e+00, 1.9960749149322510e-01, -3.4049253910779953e-02));
axis.push_back(Point3f(3.4442718029022217e+00, -1.3737335801124573e-01, -3.2491162419319153e-02));
axis.push_back(Point3f(3.1073021888732910e+00, -1.4059776067733765e-01, -2.9306696727871895e-02));
axis.push_back(Point3f(3.5181791782379150e+00, 2.0744231343269348e-01, -3.7414714694023132e-02));
axis.push_back(Point3f(3.8551650047302246e+00, 2.0967239141464233e-01, -3.9523418992757797e-02));
axis.push_back(Point3f(3.8574001789093018e+00, -1.2731924653053284e-01, -3.8719505071640015e-02));
axis.push_back(Point3f(3.5204143524169922e+00, -1.2954932451248169e-01, -3.6610800772905350e-02));
axis.push_back(Point3f(-1.7288476228713989e-01, 5.8759176731109619e-01, -5.1794531755149364e-03));
axis.push_back(Point3f(1.6411209106445312e-01, 5.8904844522476196e-01, -5.0482382066547871e-03));
axis.push_back(Point3f(1.6556793451309204e-01, 2.5205773115158081e-01, -3.0110008083283901e-03));
axis.push_back(Point3f(-1.7142891883850098e-01, 2.5060105323791504e-01, -3.1422160100191832e-03));
axis.push_back(Point3f(2.3041751980781555e-01, 6.0689312219619751e-01, -1.3651641085743904e-02));
axis.push_back(Point3f(5.6740355491638184e-01, 6.0642242431640625e-01, -1.6689710319042206e-02));
axis.push_back(Point3f(5.6693577766418457e-01, 2.6942288875579834e-01, -1.6365567222237587e-02));
axis.push_back(Point3f(2.2994974255561829e-01, 2.6989358663558960e-01, -1.3327498920261860e-02));
axis.push_back(Point3f(6.4472639560699463e-01, 5.8837163448333740e-01, -9.9883228540420532e-03));
axis.push_back(Point3f(9.8171508312225342e-01, 5.8956944942474365e-01, -1.2473850511014462e-02));
axis.push_back(Point3f(9.8291909694671631e-01, 2.5257262587547302e-01, -1.1641323566436768e-02));
axis.push_back(Point3f(6.4593040943145752e-01, 2.5137481093406677e-01, -9.1557959094643593e-03));
axis.push_back(Point3f(1.0546441078186035e+00, 5.8453100919723511e-01, -1.0812859982252121e-02));
axis.push_back(Point3f(1.3916300535202026e+00, 5.8580321073532104e-01, -1.3621388934552670e-02));
axis.push_back(Point3f(1.3929045200347900e+00, 2.4880568683147430e-01, -1.3355987146496773e-02));
axis.push_back(Point3f(1.0559185743331909e+00, 2.4753351509571075e-01, -1.0547458194196224e-02));
axis.push_back(Point3f(1.4633282423019409e+00, 5.8825415372848511e-01, -1.4365269802510738e-02));
axis.push_back(Point3f(1.8003239631652832e+00, 5.8947116136550903e-01, -1.5571499243378639e-02));
axis.push_back(Point3f(1.8015412092208862e+00, 2.5247335433959961e-01, -1.5506078489124775e-02));
axis.push_back(Point3f(1.4645454883575439e+00, 2.5125634670257568e-01, -1.4299849048256874e-02));
axis.push_back(Point3f(1.8722270727157593e+00, 5.8949387073516846e-01, -1.6279589384794235e-02));
axis.push_back(Point3f(2.2092187404632568e+00, 5.9058886766433716e-01, -1.8386986106634140e-02));
axis.push_back(Point3f(2.2103223800659180e+00, 2.5359350442886353e-01, -1.6993977129459381e-02));
axis.push_back(Point3f(1.8733308315277100e+00, 2.5249850749969482e-01, -1.4886581338942051e-02));
axis.push_back(Point3f(2.2813224792480469e+00, 5.9365952014923096e-01, -2.0738698542118073e-02));
axis.push_back(Point3f(2.6183142662048340e+00, 5.9454309940338135e-01, -2.2897880524396896e-02));
axis.push_back(Point3f(2.6192111968994141e+00, 2.5755065679550171e-01, -2.0827259868383408e-02));
axis.push_back(Point3f(2.2822194099426270e+00, 2.5666707754135132e-01, -1.8668077886104584e-02));
axis.push_back(Point3f(2.6918728351593018e+00, 6.0313165187835693e-01, -2.6091016829013824e-02));
axis.push_back(Point3f(3.0288629531860352e+00, 6.0407882928848267e-01, -2.8462318703532219e-02));
axis.push_back(Point3f(3.0298197269439697e+00, 2.6708287000656128e-01, -2.7103226631879807e-02));
axis.push_back(Point3f(2.6928296089172363e+00, 2.6613569259643555e-01, -2.4731924757361412e-02));
axis.push_back(Point3f(3.1021716594696045e+00, 6.0912829637527466e-01, -3.0640732496976852e-02));
axis.push_back(Point3f(3.4391560554504395e+00, 6.1097890138626099e-01, -3.3283811062574387e-02));
axis.push_back(Point3f(3.4410116672515869e+00, 2.7398461103439331e-01, -3.2647673040628433e-02));
axis.push_back(Point3f(3.1040272712707520e+00, 2.7213400602340698e-01, -3.0004594475030899e-02));
axis.push_back(Point3f(3.5179045200347900e+00, 6.2527948617935181e-01, -3.9177849888801575e-02));
axis.push_back(Point3f(3.8548958301544189e+00, 6.2685716152191162e-01, -4.1021160781383514e-02));
axis.push_back(Point3f(3.8564736843109131e+00, 2.8986084461212158e-01, -4.0951058268547058e-02));
axis.push_back(Point3f(3.5194823741912842e+00, 2.8828316926956177e-01, -3.9107747375965118e-02));
axis.push_back(Point3f(-1.7209273576736450e-01, 9.8850291967391968e-01, -1.5779903624206781e-03));
axis.push_back(Point3f(1.6489517688751221e-01, 9.9133694171905518e-01, -1.2400134000927210e-03));
axis.push_back(Point3f(1.6772690415382385e-01, 6.5435630083084106e-01, 1.0089552961289883e-03));
axis.push_back(Point3f(-1.6926100850105286e-01, 6.5152227878570557e-01, 6.7097839200869203e-04));
axis.push_back(Point3f(2.3108242452144623e-01, 1.0117944478988647e+00, -9.6378494054079056e-03));
axis.push_back(Point3f(5.6806731224060059e-01, 1.0126867294311523e+00, -1.2701363302767277e-02));
axis.push_back(Point3f(5.6896114349365234e-01, 6.7568802833557129e-01, -1.2538621202111244e-02));
axis.push_back(Point3f(2.3197625577449799e-01, 6.7479568719863892e-01, -9.4751073047518730e-03));
axis.push_back(Point3f(6.4327770471572876e-01, 1.0006061792373657e+00, -9.6499156206846237e-03));
axis.push_back(Point3f(9.8027110099792480e-01, 1.0018992424011230e+00, -1.1318380944430828e-02));
axis.push_back(Point3f(9.8156708478927612e-01, 6.6490226984024048e-01, -1.0729463770985603e-02));
axis.push_back(Point3f(6.4457368850708008e-01, 6.6360920667648315e-01, -9.0609984472393990e-03));
axis.push_back(Point3f(1.0526446104049683e+00, 9.9968093633651733e-01, -1.1468920856714249e-02));
axis.push_back(Point3f(1.3896282911300659e+00, 1.0007882118225098e+00, -1.4595852233469486e-02));
axis.push_back(Point3f(1.3907440900802612e+00, 6.6379123926162720e-01, -1.3685107231140137e-02));
axis.push_back(Point3f(1.0537604093551636e+00, 6.6268402338027954e-01, -1.0558175854384899e-02));
axis.push_back(Point3f(1.4615941047668457e+00, 1.0054082870483398e+00, -1.6643222421407700e-02));
axis.push_back(Point3f(1.7985901832580566e+00, 1.0061377286911011e+00, -1.8089238554239273e-02));
axis.push_back(Point3f(1.7993266582489014e+00, 6.6914260387420654e-01, -1.6437422484159470e-02));
axis.push_back(Point3f(1.4623305797576904e+00, 6.6841316223144531e-01, -1.4991406351327896e-02));
axis.push_back(Point3f(1.8702523708343506e+00, 1.0032976865768433e+00, -1.8427239730954170e-02));
axis.push_back(Point3f(2.2072377204895020e+00, 1.0043305158615112e+00, -2.1391108632087708e-02));
axis.push_back(Point3f(2.2082946300506592e+00, 6.6734337806701660e-01, -1.8645619973540306e-02));
axis.push_back(Point3f(1.8713092803955078e+00, 6.6631054878234863e-01, -1.5681751072406769e-02));
axis.push_back(Point3f(2.2797391414642334e+00, 1.0078018903732300e+00, -2.3463545367121696e-02));
axis.push_back(Point3f(2.6167349815368652e+00, 1.0086017847061157e+00, -2.4931490421295166e-02));
axis.push_back(Point3f(2.6175458431243896e+00, 6.7161238193511963e-01, -2.2379426285624504e-02));
axis.push_back(Point3f(2.2805500030517578e+00, 6.7081248760223389e-01, -2.0911481231451035e-02));
axis.push_back(Point3f(2.6919441223144531e+00, 1.0245735645294189e+00, -3.2081160694360733e-02));
axis.push_back(Point3f(3.0289404392242432e+00, 1.0252203941345215e+00, -3.3567186444997787e-02));
axis.push_back(Point3f(3.0296001434326172e+00, 6.8823355436325073e-01, -3.0656795948743820e-02));
axis.push_back(Point3f(2.6926038265228271e+00, 6.8758666515350342e-01, -2.9170770198106766e-02));
axis.push_back(Point3f(3.1008057594299316e+00, 1.0221266746520996e+00, -3.2041560858488083e-02));
axis.push_back(Point3f(3.4378008842468262e+00, 1.0229691267013550e+00, -3.3626008778810501e-02));
axis.push_back(Point3f(3.4386529922485352e+00, 6.8597626686096191e-01, -3.1595949083566666e-02));
axis.push_back(Point3f(3.1016578674316406e+00, 6.8513381481170654e-01, -3.0011501163244247e-02));
axis.push_back(Point3f(3.5181488990783691e+00, 1.0449517965316772e+00, -4.1926365345716476e-02));
axis.push_back(Point3f(3.8551383018493652e+00, 1.0465228557586670e+00, -4.4073227792978287e-02));
axis.push_back(Point3f(3.8567171096801758e+00, 7.0952868461608887e-01, -4.2861860245466232e-02));
axis.push_back(Point3f(3.5197277069091797e+00, 7.0795768499374390e-01, -4.0714997798204422e-02));
axis.push_back(Point3f(-1.8690973520278931e-01, 1.4349313974380493e+00, -1.7469346523284912e-02));
axis.push_back(Point3f(1.5007278323173523e-01, 1.4380048513412476e+00, -1.9001685082912445e-02));
axis.push_back(Point3f(1.5316531062126160e-01, 1.1010454893112183e+00, -1.4777693897485733e-02));
axis.push_back(Point3f(-1.8381720781326294e-01, 1.0979720354080200e+00, -1.3245354406535625e-02));
axis.push_back(Point3f(2.2570830583572388e-01, 1.4304934740066528e+00, -1.5912154689431190e-02));
axis.push_back(Point3f(5.6269222497940063e-01, 1.4330761432647705e+00, -1.7953991889953613e-02));
axis.push_back(Point3f(5.6529033184051514e-01, 1.0960956811904907e+00, -1.5419008210301399e-02));
axis.push_back(Point3f(2.2830641269683838e-01, 1.0935130119323730e+00, -1.3377171009778976e-02));
axis.push_back(Point3f(6.3977527618408203e-01, 1.4165247678756714e+00, -1.2272411957383156e-02));
axis.push_back(Point3f(9.7676414251327515e-01, 1.4186506271362305e+00, -1.4002894982695580e-02));
axis.push_back(Point3f(9.7889900207519531e-01, 1.0816618204116821e+00, -1.2262778356671333e-02));
axis.push_back(Point3f(6.4191013574600220e-01, 1.0795359611511230e+00, -1.0532295331358910e-02));
axis.push_back(Point3f(1.0499970912933350e+00, 1.4130892753601074e+00, -1.2655984610319138e-02));
axis.push_back(Point3f(1.3869729042053223e+00, 1.4151272773742676e+00, -1.6143925487995148e-02));
axis.push_back(Point3f(1.3890349864959717e+00, 1.0781416893005371e+00, -1.3813633471727371e-02));
axis.push_back(Point3f(1.0520591735839844e+00, 1.0761036872863770e+00, -1.0325692594051361e-02));
axis.push_back(Point3f(1.4590972661972046e+00, 1.4114594459533691e+00, -1.5034011565148830e-02));
axis.push_back(Point3f(1.7960708141326904e+00, 1.4135713577270508e+00, -1.8686482682824135e-02));
axis.push_back(Point3f(1.7982145547866821e+00, 1.0765910148620605e+00, -1.5756392851471901e-02));
axis.push_back(Point3f(1.4612410068511963e+00, 1.0744791030883789e+00, -1.2103922665119171e-02));
axis.push_back(Point3f(1.8688222169876099e+00, 1.4218015670776367e+00, -2.2955540567636490e-02));
axis.push_back(Point3f(2.2057986259460449e+00, 1.4226604700088501e+00, -2.6837682351469994e-02));
axis.push_back(Point3f(2.2067027091979980e+00, 1.0856842994689941e+00, -2.2933050990104675e-02));
axis.push_back(Point3f(1.8697260618209839e+00, 1.0848253965377808e+00, -1.9050909206271172e-02));
axis.push_back(Point3f(2.2784695625305176e+00, 1.4261851310729980e+00, -2.8994960710406303e-02));
axis.push_back(Point3f(2.6154632568359375e+00, 1.4268000125885010e+00, -3.0935274437069893e-02));
axis.push_back(Point3f(2.6161050796508789e+00, 1.0898332595825195e+00, -2.6245305314660072e-02));
axis.push_back(Point3f(2.2791113853454590e+00, 1.0892183780670166e+00, -2.4304991587996483e-02));
axis.push_back(Point3f(2.6898522377014160e+00, 1.4392693042755127e+00, -3.5140261054039001e-02));
axis.push_back(Point3f(3.0268461704254150e+00, 1.4404491186141968e+00, -3.6781176924705505e-02));
axis.push_back(Point3f(3.0280427932739258e+00, 1.1034684181213379e+00, -3.3371903002262115e-02));
axis.push_back(Point3f(2.6910488605499268e+00, 1.1022886037826538e+00, -3.1730987131595612e-02));
axis.push_back(Point3f(3.1006178855895996e+00, 1.4419562816619873e+00, -3.8106266409158707e-02));
axis.push_back(Point3f(3.4376158714294434e+00, 1.4426196813583374e+00, -3.9018034934997559e-02));
axis.push_back(Point3f(3.4382901191711426e+00, 1.1056449413299561e+00, -3.4944649785757065e-02));
axis.push_back(Point3f(3.1012921333312988e+00, 1.1049815416336060e+00, -3.4032881259918213e-02));
axis.push_back(Point3f(3.5145010948181152e+00, 1.4546076059341431e+00, -4.2183920741081238e-02));
axis.push_back(Point3f(3.8514935970306396e+00, 1.4563944339752197e+00, -4.3571799993515015e-02));
axis.push_back(Point3f(3.8532896041870117e+00, 1.1194068193435669e+00, -4.1298598051071167e-02));
axis.push_back(Point3f(3.5162971019744873e+00, 1.1176199913024902e+00, -3.9910718798637390e-02));
axis.push_back(Point3f(-1.9069427251815796e-01, 1.8625795841217041e+00, -1.6088413074612617e-02));
axis.push_back(Point3f(1.4628395438194275e-01, 1.8634387254714966e+00, -1.9822210073471069e-02));
axis.push_back(Point3f(1.4710557460784912e-01, 1.5264568328857422e+00, -2.3210266605019569e-02));
axis.push_back(Point3f(-1.8987265229225159e-01, 1.5255976915359497e+00, -1.9476469606161118e-02));
axis.push_back(Point3f(2.2714400291442871e-01, 1.8408596515655518e+00, -1.2855038978159428e-02));
axis.push_back(Point3f(5.6414365768432617e-01, 1.8412818908691406e+00, -1.3048517517745495e-02));
axis.push_back(Point3f(5.6456518173217773e-01, 1.5042846202850342e+00, -1.4329125173389912e-02));
axis.push_back(Point3f(2.2756549715995789e-01, 1.5038623809814453e+00, -1.4135646633803844e-02));
axis.push_back(Point3f(6.3799321651458740e-01, 1.8347859382629395e+00, -1.1603215709328651e-02));
axis.push_back(Point3f(9.7498786449432373e-01, 1.8359427452087402e+00, -1.3115568086504936e-02));
axis.push_back(Point3f(9.7614014148712158e-01, 1.4989461898803711e+00, -1.4137148857116699e-02));
axis.push_back(Point3f(6.3914549350738525e-01, 1.4977893829345703e+00, -1.2624796479940414e-02));
axis.push_back(Point3f(1.0474162101745605e+00, 1.8353484869003296e+00, -1.2948554009199142e-02));
axis.push_back(Point3f(1.3844008445739746e+00, 1.8365405797958374e+00, -1.5942402184009552e-02));
axis.push_back(Point3f(1.3855798244476318e+00, 1.4995459318161011e+00, -1.7433660104870796e-02));
axis.push_back(Point3f(1.0485951900482178e+00, 1.4983538389205933e+00, -1.4439810998737812e-02));
axis.push_back(Point3f(1.4563553333282471e+00, 1.8384151458740234e+00, -1.7655622214078903e-02));
axis.push_back(Point3f(1.7933298349380493e+00, 1.8400490283966064e+00, -2.1464392542839050e-02));
axis.push_back(Point3f(1.7949595451354980e+00, 1.5030531883239746e+00, -2.1839004009962082e-02));
axis.push_back(Point3f(1.4579850435256958e+00, 1.5014193058013916e+00, -1.8030233681201935e-02));
axis.push_back(Point3f(1.8658139705657959e+00, 1.8315508365631104e+00, -1.9919553771615028e-02));
axis.push_back(Point3f(2.2027769088745117e+00, 1.8333148956298828e+00, -2.4591671302914619e-02));
axis.push_back(Point3f(2.2045338153839111e+00, 1.4963197708129883e+00, -2.5116378441452980e-02));
axis.push_back(Point3f(1.8675708770751953e+00, 1.4945557117462158e+00, -2.0444260910153389e-02));
axis.push_back(Point3f(2.2757976055145264e+00, 1.8391783237457275e+00, -2.7503840625286102e-02));
axis.push_back(Point3f(2.6127901077270508e+00, 1.8402836322784424e+00, -2.9449217021465302e-02));
axis.push_back(Point3f(2.6138932704925537e+00, 1.5032856464385986e+00, -2.9821056872606277e-02));
axis.push_back(Point3f(2.2769007682800293e+00, 1.5021803379058838e+00, -2.7875680476427078e-02));
axis.push_back(Point3f(2.6859660148620605e+00, 1.8479379415512085e+00, -3.0908318236470222e-02));
axis.push_back(Point3f(3.0229606628417969e+00, 1.8493301868438721e+00, -3.2219007611274719e-02));
axis.push_back(Point3f(3.0243487358093262e+00, 1.5123347043991089e+00, -3.3276554197072983e-02));
axis.push_back(Point3f(2.6873540878295898e+00, 1.5109424591064453e+00, -3.1965866684913635e-02));
axis.push_back(Point3f(3.0954115390777588e+00, 1.8481513261795044e+00, -3.2357726246118546e-02));
axis.push_back(Point3f(3.4324009418487549e+00, 1.8496986627578735e+00, -3.4535426646471024e-02));
axis.push_back(Point3f(3.4339435100555420e+00, 1.5127030611038208e+00, -3.5287175327539444e-02));
axis.push_back(Point3f(3.0969541072845459e+00, 1.5111557245254517e+00, -3.3109474927186966e-02));
axis.push_back(Point3f(3.5111908912658691e+00, 1.8701969385147095e+00, -4.0973342955112457e-02));
axis.push_back(Point3f(3.8481795787811279e+00, 1.8725779056549072e+00, -4.2394127696752548e-02));
axis.push_back(Point3f(3.8505587577819824e+00, 1.5355864763259888e+00, -4.2787358164787292e-02));
axis.push_back(Point3f(3.5135700702667236e+00, 1.5332055091857910e+00, -4.1366573423147202e-02));





        while(1==1)
        {
            if (!vreader.isOpened()) throw std::runtime_error("Could not open input");
            //read input image(or first image from video)
            vreader>>InImage;

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

                        vector<Point2f> imagePoints,imagePoints2,imagePoints3;
                       // vector<Point3f> axis,axis2,axis3;

                          //on one ground marker
                        // axis.push_back(Point3f(1.058,0.66,0.337));
                        // axis.push_back(Point3f(1.395,0.66,0.337));
                        // axis.push_back(Point3f(1.395,0.997,0.337));
                        // axis.push_back(Point3f(1.058,0.997,0.337));
                        // axis.push_back(Point3f(1.058,0.66,0));
                        // axis.push_back(Point3f(1.395,0.66,0));
                        // axis.push_back(Point3f(1.395,0.997,0));
                        // axis.push_back(Point3f(1.058,0.997,0));

                        // axis2.push_back(Point3f(1.058,0.33,0.337));
                        // axis2.push_back(Point3f(1.395,0.33,0.337));
                        // axis2.push_back(Point3f(1.395,0.667,0.337));
                        // axis2.push_back(Point3f(1.058,0.667,0.337));
                        // axis2.push_back(Point3f(1.058,0.33,0));
                        // axis2.push_back(Point3f(1.395,0.33,0));
                        // axis2.push_back(Point3f(1.395,0.667,0));
                        // axis2.push_back(Point3f(1.058,0.667,0));

                        // axis3.push_back(Point3f(1.058,0.0,0.337));
                        // axis3.push_back(Point3f(1.395,0.0,0.337));
                        // axis3.push_back(Point3f(1.395,0.337,0.337));
                        // axis3.push_back(Point3f(1.058,0.337,0.337));
                        // axis3.push_back(Point3f(1.058,0.0,0));
                        // axis3.push_back(Point3f(1.395,0.0,0));
                        // axis3.push_back(Point3f(1.395,0.337,0));
                        // axis3.push_back(Point3f(1.058,0.337,0));

                       

                        projectPoints(axis, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints);
                         // projectPoints(axis2, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints2);
                         //  projectPoints(axis3, MSPoseTracker.getRvec(),MSPoseTracker.getTvec(), CamParam.CameraMatrix, CamParam.Distorsion,imagePoints3);
                        Mat res;
                        InImage.copyTo(res);


                        //draw(in_copy,imagePoints);
                        //draw(in_copy,imagePoints2);
                        //cv::line(res,imagePoints[0],imagePoints[1],(122,135,145),5,8,0);
                        
                        for (int i = 0; i < imagePoints.size();)
                        {
                        	cv::line(res,imagePoints[i],imagePoints[i+1],cvScalar(0,0,255),1,8,0);
                        	cv::line(res,imagePoints[i+1],imagePoints[i+2],cvScalar(0,0,255),1,8,0);
                        	cv::line(res,imagePoints[i+2],imagePoints[i+3],cvScalar(0,0,255),1,8,0);
                        	cv::line(res,imagePoints[i+3],imagePoints[i],cvScalar(0,0,255),1,8,0);
                        	i+=4;
                        }
                        vcap<<res;
                        imshow("virtual pyramid",res);
                       // cout<<MSPoseTracker.getRTMatrix();
                        cout<<"ROT:"<<MSPoseTracker.getRvec()<<endl;
                        aruco::CvDrawingUtils::draw3dAxis(InImage,CamParam,MSPoseTracker.getRvec(),MSPoseTracker.getTvec(),TheMarkerMapConfig[0].getMarkerSize()*2);
                        marker_size=TheMarkerMapConfig[0].getMarkerSize()*2;
                    

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
        if (argc >= 6) cv::imwrite(argv[5], InImage);
        }


    } catch (std::exception &ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
}