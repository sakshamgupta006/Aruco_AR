#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
using namespace std;
using namespace cv;

void readFromFile(string sfile) throw(cv::Exception) {
    try {
        cv::FileStorage fs(sfile, cv::FileStorage::READ);
        readFromFile(fs);
    } catch (std::exception &ex) {
        throw cv::Exception(81818, "MarkerMap::readFromFile", ex.what() + string(" file=)") + sfile, __FILE__, __LINE__);
    }
}


/**Reads board info from a file
*/
void readFromFile(cv::FileStorage &fs) throw(cv::Exception) {
    int aux = 0;
    // look for the nmarkers
    if (fs["aruco_bc_nmarkers"].name() != "aruco_bc_nmarkers")
        throw cv::Exception(81818, "MarkerMap::readFromFile", "invalid file type", __FILE__, __LINE__);
    fs["aruco_bc_nmarkers"] >> aux;
    resize(aux);
    fs["aruco_bc_mInfoType"] >> mInfoType;
    cv::FileNode markers = fs["aruco_bc_markers"];
    int i = 0;
    for (FileNodeIterator it = markers.begin(); it != markers.end(); ++it, i++) {
        at(i).id = (*it)["id"];
        FileNode FnCorners = (*it)["corners"];
        for (FileNodeIterator itc = FnCorners.begin(); itc != FnCorners.end(); ++itc) {
            vector< float > coordinates3d;
            (*itc) >> coordinates3d;
            if (coordinates3d.size() != 3)
                throw cv::Exception(81818, "MarkerMap::readFromFile", "invalid file type 3", __FILE__, __LINE__);
            cv::Point3f point(coordinates3d[0], coordinates3d[1], coordinates3d[2]);
            at(i).push_back(point);
            cout<<point<<endl;
        }
    }

    if (fs["aruco_bc_dict"].name()=="aruco_bc_dict")
     fs["aruco_bc_dict"] >> dictionary;

}


int main{
	readFromFile("map_cap1.yml");


	return 0;
}

