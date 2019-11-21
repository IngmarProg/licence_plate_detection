// **************************************************************************************************
// Licence https://www.gnu.org/licenses/gpl-3.0.html
// **************************************************************************************************
#include "stdafx.h"
#include <windows.h>
#include "../test_files.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "text.hpp"
#include "erfilter.hpp"
#include "ocr.hpp"
#include <iostream>
#include <chrono>
#include <process.h>


#include <codecvt> // for wstring_convert
#include <locale>  // for codecvt_byname
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>

#if PLATEMODE == 2
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "region.h"
#include "agglomerative_clustering.h"
#include "stopping_rule.h"
#include "utils.h"
#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <stdio.h>
#include <atomic> 


#ifdef _DEBUG  
/*
#pragma comment(lib, "opencv_core300d.lib")     
#pragma comment(lib, "opencv_highgui300d.lib")  
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_text300d.lib")
#pragma comment(lib, "opencv_features2d300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")
*/
#else     
#pragma comment(lib, "opencv_core300.lib")     
#pragma comment(lib, "opencv_highgui300.lib")  
#pragma comment(lib, "opencv_imgcodecs300.lib")  
#pragma comment(lib, "opencv_text300.lib")
#pragma comment(lib, "opencv_features2d300.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")
#endif      


using namespace std;
using namespace cv;

/* ************************************************************************** */

// using namespace cv::text;
Ptr< text::ERFilter> er_filter1;
Ptr< text::ERFilter> er_filter2;


int  _resize_coef = 4; // Mitu korda leitud numbrit suurendada
bool _allowcolorfilter = false; // Vaatab, et leitud märgis on ikka valge ja musta seos olemas; samas välismaa numbritega ei tööta, seal võib olla kollane ala valge asemel või sinine diplomaatidel

// Käsurea parameetrid
#ifdef TESTING
bool _showfoundareas = true;
#else
bool _showfoundareas = false; // Kas kuvatakse kõik leitud regioonid
#endif 
bool _writeareastofile = false; // Kas kirjutame leitud alad faili 
bool _videoprocessing = false;
bool _countour_fix = false; // Kas proovime täita tühikuid leitud tähtedes
bool _clear_countour = true; // Kontuurides igasugu müra, püüame selle eemalda
string _outputdir = "";
string _videofile = "";
// teeme pildi väiksemaks
int _perctop = 0;
int _percbottom = 0;


// Kiudude loend, mis videot töötlevad
// std::atomic<double> aPass;
std::atomic<int> __threadcount;

/* ************************************************************************** */
// DEBUG
/* ************************************************************************** */

int _runindex = 0;

/* ************************************************************************** */

int ColorAsRGB(int Red, int Green, int Blue) {
	return (Red * 65536) + (Green * 256) + Blue;
	// return (int)(Red << 16 | Green << 8 | Blue << 4);
}

/* ************************************************************************** */

void groups_draw(Mat &src, vector< Rect> &groups) {
	for (int i = (int)groups.size() - 1; i >= 0; i--) {
		if (src.type() == CV_8UC3)
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(0, 255, 255), 3, 8);
		else
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(255), 3, 8);
	}
}

/* ************************************************************************** */
// http://answers.opencv.org/question/74229/createerfilternm1-parameter/
// https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/textdetection.cpp
// Kahjuks ootab värvilist pilti !!!

bool detectRealTextualAreas(Mat img_region) {
	vector< Mat> channels;
	text::computeNMChannels(img_region, channels);
	return true;
	int cn = (int)channels.size();
	// Append negative channels to detect ER- (bright regions over dark background)
	for (int c = 0; c < cn - 1; c++) {
		channels.push_back(255 - channels[c]);
	}

	vector< vector< text::ERStat> > regions(channels.size());
	for (int c = 0; c< (int)channels.size(); c++) {
		er_filter1->run(channels[c], regions[c]);
		er_filter2->run(channels[c], regions[c]);
	}

	vector< vector< Vec2i> > region_groups;
	vector< Rect> groups_boxes;
	text::erGrouping(img_region, channels, regions, region_groups, groups_boxes, text::ERGROUPING_ORIENTATION_HORIZ);


	groups_draw(img_region, groups_boxes);
	imshow("grouping", img_region);
	waitKey(0);

#ifdef  DEBUG
	groups_draw(img_region, groups_boxes);
	imshow("grouping", img_region);
	waitKey(0);
#endif //  DEBUG

	regions.clear();
	int grp_cnt = groups_boxes.size();
	if (!groups_boxes.empty()) {
		groups_boxes.clear();
	}
	return (bool)grp_cnt > 0;
}

/* ************************************************************************** */

struct sortbyX {
	bool operator () (const Rect & p1, const Rect & p2) {
		return p1.x < p2.x;
	}
};

/* ************************************************************************** */

struct sortbyY {
	bool operator () (const Rect & p1, const Rect & p2) {
		return p1.y < p2.y;
	}
};

/* ************************************************************************** */

unsigned getUniqueColor_v1(const Scalar v) {	
	int v2 = v.val[2], v1 = v.val[1], v0 = v.val[0];
	return ((v2 & 0xff) << 16) + ((v2 & 0xff) << 8) + (v0 & 0xff);
}

/* ************************************************************************** */

// string filename = parser.get<string>(0);
// https://stackoverflow.com/questions/32067159/is-there-a-direct-way-to-get-a-unique-value-representing-rgb-color-in-opencv-c


/* ************************************************************************** */

unsigned getUniqueColor_v1(const Vec3b& v) {
	return ((v[2] & 0xff) << 16) + ((v[1] & 0xff) << 8) + (v[0] & 0xff);
}

/* ************************************************************************** */

unsigned getUniqueColor_v2(const Vec3b& v) {
	return 0x00ffffff & *((unsigned*)(v.val));
}

/* ************************************************************************** */

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

/* ************************************************************************** */

bool __compare(Rect l, Rect r) {
	//if (l.x == r.x) return true;
	//return (l.x < r.x);
	return true;
}

/* ************************************************************************** */

Mat eqHistogram(Mat image) {
	Mat image_hsv, image_eq;
	vector<Mat> hsvSplit;

	if (image.channels() == 1) {
		equalizeHist(image, image_eq);

	}
	else if (image.channels() == 3) {
		cvtColor(image, image_hsv, CV_BGR2HSV);
		split(image_hsv, hsvSplit);

		// img_resized = image.clone();
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, image_hsv);
		cvtColor(image_hsv, image_eq, CV_HSV2BGR);
	}

	return image_eq;
}

/* ************************************************************************** */
// Vaatame, et numbri ala liiga tume ei oleks, kui on, siis proovime heledamaks teha ehk leiame miskit
/* ************************************************************************** */

bool baseRegionTooDark(Mat img_region) {
	Mat greycharimg;
	cvtColor(img_region, greycharimg, CV_BGR2GRAY);
	blur(greycharimg, greycharimg, Size(3, 3));
	int brightcolor = 0, darkcolor = 0;
	// kas ei peaks pigem keskelt kontrollima, saame täpsemad tulemused ?!?
	for (int y = 0; y < greycharimg.rows; y++) {
		for (int x = 0; x < greycharimg.cols; x++) {
			Scalar colour = greycharimg.at<uchar>(Point(x, y));
			// cout << colour.val[0] << endl;
			if (colour.val[0] > 135) { // kas 135 on piisav ?!?!
				brightcolor++;
			}
			else {
				darkcolor++;
			}
		}
	}

	bool dark = (bool)((float)brightcolor * 2.1 < (float)darkcolor);
	return dark;
}

/* ************************************************************************** */
// Kas pildid sarnased, kiire meetod, aga ka vearohke
/* ************************************************************************** */

// https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
double getSimilarity(const Mat A, const Mat B) {
	if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
		// Calculate the L2 relative error between images.
		double errorL2 = norm(A, B, CV_L2);
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / (double)(A.rows * A.cols);
		return similarity;
	}
	else {
		//Images have a different size
		return 100000000.0;  // Return a bad value
	}
}

/* ************************************************************************** */
/* Eemaldame pildist imelikud kontuurid
/* ************************************************************************** */

Mat contoursFix(Mat imgcat) {	
	// _clear_countour
	Mat dest;
	Mat src = imgcat.clone();
	if (_clear_countour) {
		vector< Vec4i > nrhierarchy;
		vector< vector <Point> > nrcontours;
		Mat rez = Mat::zeros(src.size(), CV_8UC1);
		// rez = Scalar(255, 255, 255); 
		// OCR ei tööta vaid heledaga hästi
		// rez = Scalar(250, 250, 250);

		rez = Scalar(255, 255, 255);
		findContours(src, nrcontours, nrhierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		for (size_t i = 0; i < nrcontours.size(); i++)
			if (contourArea(nrcontours[i]) > 250) { // 128; 256
				Scalar color = Scalar(0, 0, 0);
				if (i > 0) {
					// https://stackoverflow.com/questions/1716274/fill-the-holes-in-opencv
					drawContours(rez, nrcontours, (int)i, color, 1, 1, nrhierarchy, 0, Point());
				}				
			}

		
		Mat x1, x2, x3;
		
		x1 = src;
		x2 = rez;
		x3 = Mat::zeros(src.size(), CV_8UC1);		
		bitwise_and(x1, x2, x3);
		dest = x3;				
	}
	

	if (_countour_fix) {
		// Suht segane värk, aga ma üritan siin ära kaotada tühjuse osade tähtede sees, ntx D täht on jäänud pooliku kontuurida
		int morph_size = 2;
		// Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
		Mat structuringElement = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1));
		// cv::morphologyEx(imgcat, imgcart_rez, 4, structuringElement, cv::Point(1, 1), 1); huvitav tulemus
		cv::morphologyEx(src, dest, 2, structuringElement, cv::Point(1, 1), 2);
		//Mat imgcart_rez2;
		//dilate(imgcart_rez, imgcart_rez2, Mat(), Point(-1, -1), 2, 1, 1);	
		return dest;
	}
	else {
		return dest;
	}
	
}


/* ************************************************************************** */
/* Teeb hallist pildist negatiivse pildi                                      */
/* ************************************************************************** */

Mat grayImageToNegative(Mat sourceImg) {
	uchar maskcolor = 150; // ulme see, et 255 teeb tagatausta täiesti valgeks, aga siis ocr ei leia hästi. 150 halli puhul parem tulemus
	Mat conv1 = sourceImg.clone();
	Mat conv2;

	if ((conv1.type() == CV_8U) || (conv1.type() == CV_8UC1)) { // conv1.type() == CV_8UC3)
		Mat to_neg = Mat::zeros(conv1.size(), conv1.type());
		Mat sub_mat = Mat(conv1.rows, conv1.cols, conv1.type(), Scalar(1, 1, 1)) * maskcolor; // Mat::ones(img_char.size(), img_char.type()) * 255;
		subtract(sub_mat, conv1, to_neg);		
		return to_neg;
	}
	else {
		// 30.08.2017 Ingmar; minu meelest see töötas paremini kui substract			
		Mat conv3;
		Mat conv0 = conv1.clone();
		cvtColor(conv0, conv1, COLOR_BGR2GRAY);
		Mat conv2 = Mat::zeros(conv1.size(), conv1.type());
		adaptiveThreshold(conv1, conv2, maskcolor, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 0);
		// adaptiveThreshold(conv1, conv2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 0);
		
		Scalar value = {};
		copyMakeBorder(conv2, conv3, 1, 1, 1, 1, BORDER_CONSTANT, value);
		return conv3;
	}
}


/* ************************************************************************** */
// TODOTODO siia tuleks teha värvide tuvastus, liiga tume ala eemaldada
/* ************************************************************************** */

bool colorsAreBogus(Mat charimg) {		
	if (!_allowcolorfilter) {
		return false;
	}

	int relativeblack = 0, relativewhite = 0, othercolors = 0;
	for (int i = 0; i < charimg.cols; i++) {
		for (int j = 0; j < charimg.rows; j++) {	
			Scalar vpixels = charimg.at<Vec3b>(j, i);
			if ((vpixels.val[0] < 95) && (vpixels.val[1] < 95) && (vpixels.val[2] < 95)) {
				relativeblack++;
			} else
			//if ((vpixels.val[0] >= 146) && (vpixels.val[1] >= 146) && (vpixels.val[2] >= 146)) {
			if ((vpixels.val[0] >= 135) && (vpixels.val[1] >= 135) && (vpixels.val[2] >= 135)) {
				relativewhite++;
			} else {
				othercolors++;
			}
			// if inRange(hsv, Scalar(110, 50, 50), Scalar(130, 255, 255), bw);			
		}
	}

	int total = othercolors + relativeblack + relativewhite;
	if (total == 0.00) return true;
	float whiteperc = ((float)relativewhite / (float)total) * 100.00;
	float blackperc = ((float)relativeblack / (float)total) * 100.00;
	float othercolorperc = 100.00 - whiteperc - blackperc;
	// need vahemikud leitud peale pikka "mängimist"
	if (((float)othercolorperc >= 35.00) || ((float)whiteperc <= 10.00) || ((float)blackperc <= 10.00) || ((float)blackperc >= 65.00))  {	
		#ifdef COUTENABLED	
		cout << " ## Odd color pattern white " << whiteperc << " Black " << blackperc << " Other " << othercolorperc << endl;
		#endif
		return true;
	}
	
	#ifdef COUTENABLED	
	cout << " ## Region color pattern white " << whiteperc << " Black " << blackperc << " Other " << othercolorperc <<endl;
	#endif
		
	return false;
}

/* ************************************************************************** */
// Siin vaatame nr märgi heleda ja tumeda relatiivset suhet; teeme nö numbrioletuse
/* ************************************************************************** */

bool areContoursOdd(Mat charimg, bool useclahe = false) {
	Mat grayimg;
	Mat ero;
	Mat dil;
	double diff;
	bool odd;
	diff = 0.00;
	cvtColor(charimg, grayimg, CV_BGR2GRAY);	
	// teeme leitud nr, tähe paksemaks
	erode(grayimg, ero, Mat(), Point(-1, -1), 2, 1, 1);

	// liiga pisikese pildi peal diff ka suurem
	int height = charimg.size().height;
	// diff = getSimilarity(ero, dil);
	diff = getSimilarity(ero, grayimg);
	// ÄRA näpi, kui ei tea mida teed ;)
	if (height >= 19) {
		odd = !(diff >= 1.6 && diff <= 7.8); // 2.4 - 3.6
	}
	else {
		odd = !(diff >= 8.7 && diff <= 12.98);
	}
	#ifdef COUTENABLED
	cout << " -- Image DIFF " << diff << " H: " << charimg.size().height << " W: " << charimg.size().width << endl;
	#endif
	return odd; 
}

/* ************************************************************************** */

double avgHeight(vector<Rect> v) {
	if (v.size() < 1) {
		return 0.00;
	}
	else {
		vector<int> heightv;
		for (unsigned int i = 0; i < v.size(); i++) {
			heightv.push_back((int)v[i].height);
		}

		return accumulate(heightv.begin(), heightv.end(), 0.0) / heightv.size();
	}	
}

/* ************************************************************************** */

double avgWidth2(vector<Rect> v) {
	if (v.size() < 1) {
		return 0.00;
	}
	else {
		vector<int> widthv;
		for (unsigned int i = 0; i < v.size(); i++) {
			widthv.push_back((int)v[i].width);
		}

		int n = 0;
		double mean = 0.0;
		for (auto x : widthv) {
			double delta = x - mean;
			mean += delta / ++n;
		}

		return mean;
	}
}

/* ************************************************************************** */

double widthDeviation(vector<Rect> v) {	
	vector <int> cpwidth;	
	Rect rct;
	// lisame leitud kõrgused
	for (unsigned i = 0; i < v.size(); i++) {
		rct = v[i];
		cpwidth.push_back(rct.width);
	}

	double sum = std::accumulate(std::begin(cpwidth), std::end(cpwidth), 0.0);
	double m = sum / cpwidth.size();

	double accum = 0.0;
	std::for_each(std::begin(cpwidth), std::end(cpwidth), [&](const double d) {
		accum += (d - m) * (d - m);
	});

	double stdev = sqrt(accum / (cpwidth.size() - 1));
	return stdev;
}

/* ************************************************************************** */
// Ignoreerib suuri kõikumisi
/* ************************************************************************** */

double avgHeight2(vector<Rect> v) {
	if (v.size() < 1) {
		return 0.00;
	}
	else {
		vector<int> heightv;
		for (unsigned int i = 0; i < v.size(); i++) {
			heightv.push_back((int)v[i].height);
		}

		int n = 0;
		double mean = 0.0;
		for (auto x : heightv) {
			double delta = x - mean;
			mean += delta / ++n;
		}
	
		return mean;
	}
}

/* ************************************************************************** */

double heightDeviation(vector<Rect> v) {
	// Samuti käia üle ristkülikud ja kõigile omistada üks kõrgus, ntx muidu võetakse ainult O tähe sisemine ring.
	// Võtame keskmise kõrguse + -2 pikslit ja kõigile ristkülikutele omistame selle kõrguse
	// Kui deviation üle 5  px, siis mingi jama ja mitte seda kasutada !
	// https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos	
	vector <int> cpheight;
	// http://www.mathsisfun.com/data/standard-deviation.html
	Rect rct;
	// lisame leitud kõrgused
	for (unsigned i = 0; i < v.size(); i++) {
		rct = v[i];
		cpheight.push_back(rct.height);
	}


	double sum = std::accumulate(std::begin(cpheight), std::end(cpheight), 0.0);
	double m = sum / cpheight.size();

	double accum = 0.0;
	std::for_each(std::begin(cpheight), std::end(cpheight), [&](const double d) {
		accum += (d - m) * (d - m);
	});

	double stdev = sqrt(accum / (cpheight.size() - 1));
	return stdev;
}


/* ************************************************************************** */
// Proovime leida numbri keskkoha
/* ************************************************************************** */

int tryToFindMiddleIndex(vector<Rect> v, double avg_width) {
	Rect cmp1, cmp2;
	int index = -1;
	int cdiff = 0, x1 = 0, x2 = 0, sepdiff = 0;
	// AGA AGA proovime leida vahekoha; siis saame järeldada, kas esimese märgi peame tuletama;
	for (unsigned int i = 0; i < v.size() - 1; i++) {
		cmp1 = v[i];
		cmp2 = v[i + 1];
		x1 = cmp1.x + cmp1.width;
		x2 = cmp2.x;
		cdiff = abs((int)x2 - (int)x1);
		// cout << "DIFFF " << x1 << " __ " << x2 << " __ "  << cdiff << endl;
		// leidsime keskpunkti ?

		// if ((cdiff > 6) && (cdiff < 46)) {
		sepdiff = (int)cvRound(avg_width / 2.3);
		#ifdef COUTENABLED
		cout << " *** Separator diff " << sepdiff << endl;		
		#endif	

		if (cdiff > sepdiff) {
			index = i; // cmp1 index !!!
			break;
		}
	}
	return index;
}

/* ************************************************************************** */
// Vaata pilti A43.jpg millegipärast ei leia viimast nr
// C:/autonr/A49.jpg kavalam predict teha !!!
// Tagastab offseti, palju Y teljel elemendid nihkes olid
/* ************************************************************************** */

int addPredictedRects(Mat source_img, vector<Rect> &v, double avg_height, double avg_height_dev, double avg_width) {
	
	// Viimased algoritmine täiendused; siin järjest lähevad tähed, aga N ei leita, 
	// peaksime arvutama ristkülikute sammu M ja U ning jätkama sama sammu,
	// kui tekib pikem auk kahe ristküliku puhul, mis ületab N pikslit
	

	int rectcount = v.size();
	int prediction = 0, diffy = 0;
	Rect firstrect, secondrect, leftpredrect, rightpredrect;


	// Enamasti probleem, et just esimest märki ei leita !!!
	// Proovime tuletada esimese märgi
	// TODO: vaadata, et seal sees poleks mingit läbu, musta ja valge korrelatsiooni vaadata

	if ((rectcount < 8) && (rectcount >= 3)) {
		// Eemaldame liiga väikesed kujundid
		// finalRemoveOddRects(v, avg_height, avg_height_dev);
		if (v.size() < 2) return 0;

		firstrect = v[0];
		secondrect = v[1];
		// thirdrect = bgrects[2];
		// vasakult (liigub) üles paremale
		int diffx = abs((int)secondrect.x - ((int)firstrect.x + (int)firstrect.width));
		diffy = (int)secondrect.y - (int)firstrect.y;
		if (diffy < 0) {
		#ifdef COUTENABLED
			std::cout << " -- Chars left < right " << diffy << endl;
		#endif // COUTENABLED			
		}
		else if (diffy > 3) {
		#ifdef COUTENABLED
			std::cout << " -- Chars left > right " << diffy << endl;
		#endif // COUTENABLED			
		}
		else {
		#ifdef COUTENABLED
			std::cout << " -- Chars norm " << diffy << endl;
		#endif // COUTENABLED			
		}

		int  xmiddlecharidx = -1, x1 = 0, x2 = 0, cdiff = 0;
		Rect cmp1, cmp2;
		
		if (v.size() >= 3) {
			cmp1 = v[0];
			cmp2 = v[1];

			// kas siin ei saaks ka kasutada avg_height ja avg_width ?			
			if ((cmp1.height > 8) && (cmp2.width > 8)) {
				x1 = cmp1.x + cmp1.width;
				x2 = cmp2.x;
				cdiff = (int)x2 - (int)x1;

				//if (cdiff > 10) {
				int cdiff2 = (int)cvRound(avg_width);
				if ((int)cdiff > (int)cdiff2) {
					leftpredrect.x = (int)cvRound(cmp1.x + cmp1.width + 2);
					leftpredrect.y = cmp1.y - (int)cvRound(diffy * 0.8);
					leftpredrect.width = cmp1.width + 1;
					leftpredrect.height = cmp1.height;
					if ((leftpredrect.x > 0) && (leftpredrect.y > 0) && (leftpredrect.height > 5) && (leftpredrect.width > 5)) {
						v.insert(v.begin(), leftpredrect);
					}
				}
			}
		}

		xmiddlecharidx = tryToFindMiddleIndex(v, avg_width);
		if ((xmiddlecharidx > 0) && (xmiddlecharidx + 1 < 3)) { // vasakule jäi alla 3 nr kandidaadi - prediction
			leftpredrect.x = (int)cvRound(firstrect.x - firstrect.width - diffx + 2);
			leftpredrect.y = firstrect.y - (int)cvRound(diffy * 0.8);
			leftpredrect.width = firstrect.width;
			leftpredrect.height = firstrect.height;
			//leftpredrect.x = firstrect.x - firstrect.width;
			// v.push_back(leftpredrect);
			if ((leftpredrect.x > 0) && (leftpredrect.y > 0) && (leftpredrect.height > 5) && (leftpredrect.width > 5)) {
				v.insert(v.begin(), leftpredrect);
			}
		}


		xmiddlecharidx = tryToFindMiddleIndex(v, avg_width);

		// Nüüd vaatame, mida teoreertilist kandidaati meil paremale jäi
		// Aga ntx lühikestel nr, mis maasturitel pole keskmist tühimikku, mootorratta nr ja ka vanade autode nr !!!!

		int missingchars = 6 - (int)v.size();
		#ifdef COUTENABLED
		std::cout << " -- Missing right chars " << missingchars << " middle index " << xmiddlecharidx << endl;
		#endif
		
		
		// Proovime paremal poole ka märkide võimalikud asukohad leida (teoreetilised)
		if ((missingchars > 0) && (xmiddlecharidx > 0) && (xmiddlecharidx + 1 <= v.size())) {
			// rightpredrect
			if (missingchars == 1) {
				rightpredrect = v[v.size() - 1];
			}
			else {
				rightpredrect = v[xmiddlecharidx + 1];
			}
			// v[xmiddlecharidx + 1].height = 8;
			// vaatame, kas järgmine märk ehk ka olemas
			if (xmiddlecharidx + 2 <= v.size() - 1) {
				Rect nextrect = v[xmiddlecharidx + 2];
				x1 = rightpredrect.x + rightpredrect.width;
				x2 = nextrect.x;
				cdiff = (int)x2 - (int)x1;

				// mõistlik vahe, tundub normaalne märk
				if (cdiff < 5) {
					rightpredrect = v[xmiddlecharidx + 2];
				}
			}
			
			Rect rrect;
			int xbw = 0, newx = 0;
			for (unsigned j = 1; j <= missingchars; j++) {
				xbw = (int)rightpredrect.width * (j); // +1 ??				
				newx = (int)rightpredrect.x + (int)xbw + (int)cvRound(diffx - 1);
				rrect.x = newx;
				if (missingchars == 1) {
					rrect.y = v[v.size() - 1].y + diffy;
				}
				else {
					rrect.y = rightpredrect.y; // +(int)cvRound(diffy * 1.4);
				}

				rrect.width = rightpredrect.width + 3;
				rrect.height = rightpredrect.height;

				//cout << xba << "xba" << endl;
				if (rrect.x + rrect.width <= source_img.size().width)
					v.push_back(rrect);
			}
		}
	}

	return diffy;
}

/* ************************************************************************** */
// TODO

void reArrangeItems(vector<Rect> &v, int i, Rect rem_elem) {
}

/* ************************************************************************** */
// Liiga väikeste laiustega ja kõrgustega ristkülikud viskame minema
/* ************************************************************************** */

// TODO refact
void removeElementsWithOddHW(vector<Rect> &v, double avg_height, double avg_height_dev, double avg_width, double avg_width_dev) {
	int maxdiff = 42;
	int cheight = (int)cvRound(avg_height * 0.8);
	int cwidth = (int)cvRound(avg_width * 0.7);
	// Kas see liiga palju ei filtreeri ?
	//  c:\autonr\7396607t1h9e3a.jpg
	for (unsigned i = v.size(); i-- > 0; )
	if ((v[i].height < avg_height) || (v[i].height > cvRound(avg_height * 1.8))) {	
		v.erase(v.begin() + i);		
	}

	// Viskame väikesed elemendid minema
	if (avg_height > 1.5)
		for (int i = (int)v.size() - 1; i >= 0; i--)
			if ((v[i].height < cheight) || (v[i].width < cwidth)) {
				v.erase(v.begin() + i);
			}

	// Viimased analüüsid, esimese märgiga ikka mingi jama, viskame minema, kuidagit filtritest läbi saanud
	if (v.size() >= 2) {
		int x1 = 0, x2 = 0, rg = 0;
		Rect cmp1, cmp2;
		cmp1 = v[0];
		cmp2 = v[1];
		x1 = (int)cmp1.x + (int)cmp1.width;
		x2 = cmp2.x;
		rg = x2 - x1;		

		// Samuti siin, kas ei võiks olla siin mitmekordne avg_width * 1.8 ?
		if (rg > maxdiff) {
			v.erase(v.begin());
		}
	}

	// Vaatame ikkagit ka veel korra lõpumärgi üle
	if (v.size() >= 2) {
		int x1 = 0, x2 = 0, rg = 0;
		Rect cmp1, cmp2;
		int lastitemidx = v.size() - 1;
		cmp1 = v[lastitemidx - 1]; 
		cmp2 = v[lastitemidx];
		x1 = (int)cmp1.x + (int)cmp1.width; 
		x2 = (int)cmp2.x;		
		rg = abs((int)x2 - (int)x1);		 
		if (rg > 12) {		
			v.erase(v.begin() + lastitemidx);
		}
	}
}

/* ************************************************************************** */
// Ristkülikute omavaheline kaugus on täiesti absurdne, eemaldame need
/* ************************************************************************** */

void removeElementsWithOddDistance(Mat source_img, vector<Rect> &v, double avgheight, double heightdev, double avg_width, double avg_width_dev) {	
	if (source_img.size().height < 32) return; // liiga väikeste piltide puhul see algoritm teeb pige kahju, kui kasu !	

	bool processed = false;
	bool keepcleaning = false;
	Rect cmp1, cmp2;
	int k = 0, x1 = 0, x2 = 0, rg = 0;
	while (!processed) {

		keepcleaning = false;
		//for (unsigned int i = k; i < v.size(); i++) {
		// Tagurpidi parem, sest esimest märki korrigeeritakse järgmistes funktsioonides
		for (int i = (int)v.size() - 1; i >= 1; i--) { // esimese märgi jätame välja
			cmp2 = v[i];
			if (i - 1 >= 0) {
				cmp1 = v[i - 1];
				x1 = cmp1.x + cmp1.width;
				x2 = cmp2.x;
				rg = x2 - x1;

				// cout << " X1 "<< x1 << " X2 " << " RG " << rg << endl;

				// NOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTE
				// Ma ei saa aru, kuidas siin saab üldse neg väärtus teha, kui X järgi sorteeritud. 
				// Paistab, et kui märgi vahel ära võtan peaks uuesti X koordinaadid uuesti arvutama kõik
				// if (rg < 0) rg = cmp2.x - cmp1.x;

				// Kas see reg. ei võiks olla pigem keskmine laius * 1.8				
				 if (rg > cvRound(avg_width * 1.6)) { // 1.6 parim koefitsent
					Rect rem_elem = v[i];
					v.erase(v.begin() + i);					
					reArrangeItems(v, i, rem_elem);
					// pole mõtet jälle alguseset peale skännima
					k = i;
					keepcleaning = true;
					break;
				}
			}
		}

		processed = !keepcleaning;
	}
}

/* ************************************************************************** */
// Kui ristkülikud liiga üksteise peal, tuleb need eemaldada
/* ************************************************************************** */

void removeElementsWithSamePos(vector<Rect> v) {
	bool processed = false;
	bool keepcleaning = false;
	Rect cmp1, cmp2;

	while (!processed) {
		keepcleaning = false;
		for (unsigned int i = 0; i < v.size(); i++) {
			cmp1 = v[i];

			for (unsigned int j = 0; j < v.size(); j++)
				if (i != j) {
					cmp2 = v[j];
					if ((cmp1.x + cmp1.width >= cmp2.x) && (cmp1.x + cmp1.width <= cmp2.x + cmp2.width)) {
						keepcleaning = true;
						if (cmp1.width > cmp2.width) {
							v.erase(v.begin() + i);
						}
						else {
							v.erase(v.begin() + j);
						}
						break;
					}
				}

			if (keepcleaning) break;
		}
		processed = !keepcleaning;
	}
}

/* ************************************************************************** */
// Proovime mõista, mis värvusega meil tegemist
/* ************************************************************************** */

int validateImagePatterns(Mat source_img, vector<Rect> &v, bool clahe) {
	Rect textrect;
	Mat charimg;
	bool allowcolorcomp = (bool)(v.size() > 2);
	for (int i = (int)v.size() - 1; i >= 0; i--) {
		textrect = v[i];
		#ifdef COUTENABLED
		//std::cout << " W " << textrect.width << " H " << textrect.height << " " << "\n";
		#endif // COUTENABLED			
		int px, py, pwidth, pheight;
		
		px = textrect.x;
		py = textrect.y;
		pwidth = textrect.width;
		pheight = textrect.height;

		// suurendame ala 3px võrra, siis tõenäosus suurem, et saame kätte tähe				
		px = px - 3 >= 0 ? px - 3 : px;
		py = py - 3 >= 0 ? py - 3 : py;
		if (py < 0) py = 0;
		if (px < 0) px = 0;

		pwidth = textrect.width + 3; // 4
		pheight = textrect.height + 3;	
		// jama, seda ei tohi juhtuda
		int pv1 = (int)pwidth + (int)px;
		if (pv1 > (int)source_img.size().width) {
			pwidth = textrect.width;
			px = 0; //  textrect.x;
		}
		
		int pv2 = (int)pheight + (int)py;
		if (pv2 > (int)source_img.size().height) {
			pheight = textrect.height;
			py = 0; //  textrect.y;
		}

		try {
			charimg = source_img(Rect(px, py, pwidth, pheight));
		}
		catch (const Exception& e) {			
			#ifdef TESTING
			cerr << endl << pv1 << " " << pv2 << " IMG:H " << source_img.size().height << " W " << source_img.size().width <<  " ERR " << " _runindex " << _runindex << " X " << px << " Y " << py << " W " << pwidth << " H " << pheight << endl << e.what() << endl;
			std::cin.ignore();
			#endif
			return 0;
		}		
		
		// Teeme jooned peeneks ja paksuks ning võrleme piltide erinevust
		if (areContoursOdd(charimg, clahe) || (allowcolorcomp && colorsAreBogus(charimg))) {
			v.erase(v.begin() + i);
		}		
	}
	return v.size();
}

/* ************************************************************************** */
// Ntx meil leiab MSER 0, siis ta kipub sisemist ala leidma, laiendame ala. Nõrga D puhul samas, samas P tähel leiab P ülemise osa vaid, siin mõelda
/* ************************************************************************** */

void smartCharResize(Mat sourceimg, vector<Rect> &v, double avg_height, double avg_height_dev) {
	int wdelta = 4;
	if ((cvRound((float)avg_height_dev * 3.01) < (float)avg_height))
	for (unsigned int i = 0; i < v.size(); i++) 
	if (v[i].height < avg_height)  {
		int h = (int)v[i].height + (int)cvRound(avg_height_dev / 1.2);
		int y = (int)v[i].y - (int)cvRound(avg_height_dev / 2.00);
		int w = (int)v[i].width + (int)wdelta * 2;
		int x = v[i].x;

		y = y < 0 ? 0 : y;
		h = h <= sourceimg.size().height ? h : v[i].height;
		v[i].y = y;
		v[i].height = h;

		if (w + x <= sourceimg.size().width) {
			v[i].width = w;
			x = x - wdelta;
			if (x >= 0) {
				v[i].x = x;
			}
		}	
	}	
}

/* ************************************************************************** */
// TODO TODO; teha väike loogika, et neg. pildist lõikame välja tumeda ristküliku
/* ************************************************************************** */

Mat leaveOnlyNegArea(Mat negativeimg) {
	return negativeimg;
}

/* ************************************************************************** */
// Lõikame neg. pildi nii pisikeseks kui vähegi oskame, et saada parim OCR tulemus
/* ************************************************************************** */

Mat cropNegImage(Mat negativeimg, vector<Rect> v, double avg_height, double avg_dev, double avg_width) {
	// lõikame alates siit ära, saame jälle pildilt jamasid objekte vähemaks
	// kui me ei leidnud ntx 6 OCR märgi, siis võtame kogu leitud ala ja teeme mustvalgeks, samas ehk leidsime ühe märgi, lõikame sellest ülemise osa ära
	if (v.size() < 2) {
		return negativeimg;
	}

	int minx = 0;
	int miny = v[0].y;
	 
	// eelviimane !!!
	if (v[v.size() - 2].y < miny) {
		miny = v[v.size() - 2].y;
	}

	// Võtame ristkülikut natuke üles poole, et paremat tulemust saada
	miny = miny - 2 >= 0 ? miny - 2 : 0;	
	int use_height = (int)cvRound((double)avg_height + (double)avg_dev) + 2;
	use_height = use_height <= negativeimg.size().height ? use_height : negativeimg.size().height;

	// Jama selles, et alati ei pruugi me õiged koordinaate algusest leida !!!
	int dcrange = 0;
	if (v.size() <= 5) {
		int missingchars = 6 - v.size();
		dcrange = cvRound((avg_width * 0.93) * (missingchars + 1) + (5 * missingchars));
	}

	minx = v[0].x - dcrange >= 0 ? v[0].x - dcrange : 0;
	int use_width = ((int)v[v.size() - 1].x + (int)v[v.size() - 1].width - v[0].x + dcrange + 18);

	if ((int)minx + (int)use_width > negativeimg.size().width) {
		use_width = negativeimg.size().width - minx;
	}

	// imshow("temp___", negativeimg(Rect(minx, miny, use_width, use_height)));
	// cvWaitKey(0);
	return leaveOnlyNegArea(negativeimg(Rect(minx, miny, use_width, use_height)));
}

/* ************************************************************************** */
// Tumedad pixlid muudame mustaks, et OCR näeks valge ala paremini
/* ************************************************************************** */

void correctDarkColors(Mat &img_best_for_orcr) {
	//uchar untilcolor = 115; // kus alates arvame, et tume värvus veel195	
	img_best_for_orcr = eqHistogram(img_best_for_orcr);
	uchar untilcolor = 160;
	uchar chans = 1 + (img_best_for_orcr.type() >> CV_CN_SHIFT);
	for (int y = 0; y < img_best_for_orcr.rows; y++) {
		for (int x = 0; x < img_best_for_orcr.cols; x++) {
			// Scalar  colour = img_best_for_orcr.at<uchar>(Point(x, y));
			// colour[0] = 10;			
			if (chans == 3) {
				Vec3b bgrPixel = img_best_for_orcr.at<Vec3b>(y, x);
				if ((int)bgrPixel[0] <= untilcolor) {
					img_best_for_orcr.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
				}
			}
			else {
				uchar & color = img_best_for_orcr.at<uchar>(y, x);				
				if (color <= untilcolor) {
					color = (uchar)(0);
				}
			}
		}
	}				
}

/* ************************************************************************** */
// Käime pildi OCRiga üle
/* ************************************************************************** */
// https://stackoverflow.com/questions/122616/how-do-i-trim-leading-trailing-whitespace-in-a-standard-way

char *trimwhitespace(char *str)
{
	char *end;

	// Trim leading space
	while (isspace((unsigned char)*str)) str++;

	if (*str == 0)  // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	// Write new null terminator
	*(end + 1) = 0;

	return str;
}

/* ************************************************************************** */

void sendToOcrProcessing(Mat sourceimg, Mat negativeimg, vector<Rect> v, int char_diff_y, double avg_height, double avg_devition, double avg_width) {
	int deltaw = 2;
	int totalwidth = 0, maxheight = 0, currentleft = 3; //  currentleft = 0 kui 0, siis ei leita ocr poolt vasakpoolsetmärki
	for (unsigned int i = 0; i < v.size(); i++) {
		totalwidth += v[i].width + deltaw;
		if (v[i].height > maxheight) {
			maxheight = v[i].height;
		}
	}

	maxheight += 4;	
	// Kopeerime leitud märgid valgele taustale, aga leian, kõige efektiivsem oleks neid ükshaaval töödelda !
	// for (unsigned int i = 0; i < v.size(); i++) {
	// cv::Mat img_for_ocr(maxheight + 4, totalwidth + 8, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat img_for_ocr(maxheight + 4, totalwidth + 8, sourceimg.type(), cv::Scalar(0, 0, 0));
	Mat img_best_for_ocr;

	// cv::Mat img_for_orc(maxheight, totalwidth, CV_8UC3, cv::Scalar(0, 0, 0));
	for (unsigned int i = 0; i < v.size(); i++)
		try {
			Mat img_char = sourceimg(v[i]);	
			// cout << type2str(img_char.type()) << endl;
			// Teeme märgist negatiivse pildi			
			Mat img_char_neg = Mat::zeros(sourceimg.size(), sourceimg.type());
			Mat sub_mat = Mat(img_char.rows, img_char.cols, img_char.type(), Scalar(1, 1, 1)) * 255; // Mat::ones(img_char.size(), img_char.type()) * 255;
			subtract(sub_mat, img_char, img_char_neg);

			
			Rect resize = Rect(currentleft, 1, img_char_neg.size().width, img_char_neg.size().height);
			img_char_neg.copyTo(img_for_ocr(resize));

			// kas ei peaks proovima otse töödelda, võrdleme pildiga ntx andmebaasis ? method=CV_TM_SQDIFF_NORMED
			// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html

			currentleft += img_char.size().width + deltaw;
		}
		catch (const Exception& e) {
			#ifdef TESTING
			cout << "ERR: " << e.what() << endl;
			#endif	
		}
		
	// NB siin on tegemist märkidega, mille meie suutsime ise leida	
	// Mat img_for_ocr_bright;
	// img_for_orc.convertTo(img_for_ocr_bright, -1, 1, 20);	
	


	// text MSER leidis sobiva märkide arvu, eeldame et seal parim kogus tähti, samut OCR lootusetu, kui tähed viltuselt 
	if ((abs(char_diff_y) >= 3 || v.size() > 5)) {		
		img_best_for_ocr = img_for_ocr;
	}
	else {
		// maxtop
		img_best_for_ocr = cropNegImage(negativeimg, v, avg_height, avg_devition, avg_width);
	}

	// Sisuliselt kõik tumedad pixlid keerame täiesti mustaks
	correctDarkColors(img_best_for_ocr);


	/*
	Mat img_final_x_ocr_file;
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(img_best_for_ocr, img_final_x_ocr_file, MORPH_CLOSE, element); // MORPH_TOPHAT
	*/

	
	Mat img_rzlarge;
	 // cv::resize(img_best_for_ocr, img_rzlarge, cv::Size(img_best_for_ocr.cols * 4, img_best_for_ocr.rows * 4), cv::INTER_NEAREST);	
	cv::resize(img_best_for_ocr, img_rzlarge, cv::Size(img_best_for_ocr.cols * _resize_coef, img_best_for_ocr.rows * _resize_coef), cv::INTER_NEAREST);
	img_best_for_ocr = grayImageToNegative(img_rzlarge);

	// TODO: Viskame välja mingid ülisuured kontuurid, millest me aru ei saa
	img_best_for_ocr = contoursFix(img_best_for_ocr);

	// Kirjutame leitud piirkonnad faili
	if ((_outputdir != "od") && (_outputdir != "outputdir")) {
		stringstream file1;
		stringstream file2;
		file1 << _outputdir << "\\" << "raw_lp_" << _runindex << ".jpg";
		file2 << _outputdir << "\\" << "proc_lp_" << _runindex << ".jpg";
		
		imwrite(file1.str(), sourceimg);
		imwrite(file2.str(), img_best_for_ocr);
	}
		 

	if (_showfoundareas) {
		// liiga väike nr, teeme suuremaks
		stringstream wndname1;
		stringstream wndname2;

		wndname1 << "RAW area_" << _runindex;
		imshow(wndname1.str(), sourceimg);

		wndname2 << "OCR candidate_" << _runindex;
		imshow(wndname2.str(), img_best_for_ocr);
	}	
}

/* ************************************************************************** */

bool regionhasText2(Mat img_region, Mat img_negative) {
	Mat textImg;
	Mat image_clahe;

	// Liiga väike pildiala, isegi kui seal tähed ei saa me sealt täpset tulemust
	if ((img_region.cols <= 3) || (img_region.size().width < 35))  return false;

	int maxrcwidth = 0, maxrcheight = 0, pxerror = 0,  matchcnt = 0;
	float ratiotopict = 0.00, regperc = 0.00;

	//Extract MSER
	// vector<Rect> bgrects;
	bool couldbenr = false;
	Rect textrect, cmp1, cmp2;
	// vector<vector<Rect*> > bgrects;
	vector<Rect> bgrects;
	vector< vector< Point> > contours;
	vector< Rect> bboxes;


	// NB siia teha täiendus, et kui pilt liiga tume, tõstame kontrasti; ntx pilt string filename = "C:/Autonr/382640hb26e.jpg";	
	bool useclahe = baseRegionTooDark(img_region);
	if (useclahe) {
		// https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
		cv::Mat lab_image;
		cv::cvtColor(img_region, lab_image, CV_BGR2Lab);

		// Extract the L channel
		std::vector<cv::Mat> lab_planes(3);
		cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

		// apply the CLAHE algorithm to the L channel
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(4);
		cv::Mat dst;
		clahe->apply(lab_planes[0], dst);

		// Merge the the color planes back into an Lab image
		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

		// convert back to RGB
		// cv::Mat image_clahe;
		cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
		/*
		Ptr<CLAHE> clahe = createCLAHE(4, Size(8, 8));
		clahe->apply(img_region, highcontrast);
		img_region = highcontrast;
		*/		
		cvtColor(image_clahe, textImg, CV_BGR2GRAY);
	}
	else {
		cvtColor(img_region, textImg, CV_BGR2GRAY);
	}

	Ptr< MSER> mser = MSER::create(21, (int)(0.00002 * textImg.cols * textImg.rows), (int)(0.05 * textImg.cols * textImg.rows), 1, 0.7);
	mser->detectRegions(textImg, contours, bboxes);

	// *******************************************************************
	// Vähemalt 3 ristkülikut peaksid olema enam vähem sama suurusega, siis tunnistame alles tekstiks
	// *******************************************************************

	for (unsigned int i = 0; i < bboxes.size(); i++) {
		textrect = bboxes[i];
		ratiotopict = ((float)textrect.width / (float)img_region.size().width) * 100;		
		if ((ratiotopict <= 2.00) || (ratiotopict > 25.00)) continue; // > 35 ?
		
		if (maxrcwidth < textrect.width) {
			maxrcwidth = textrect.width;
		}

		if (maxrcheight < textrect.height) {
			maxrcheight = textrect.height;
		}
		// std::cout << "ratiotopict: " << ratiotopict << endl;
	}

	// *******************************************************************
	// Nüüd vaatame, kas on sarnaseid ristkülikuid, mille suurus vähemalt N % max. ristküliku omast	
	// *******************************************************************

	for (unsigned int i = 0; i < bboxes.size(); i++) {			
		textrect = bboxes[i];	
		
		//rectangle(img_region, textrect, CV_RGB(0, 255, 0));		
		regperc = ((float)textrect.width / (float)maxrcwidth) * 100.0;
		// laius peab olema max ristküliku laiusest ca 65 % ja üles
		// if ((regperc < 60.00) || (regperc > 100.00)) continue;
		if (regperc > 100.00) continue;		

		regperc = ((float)textrect.height / (float)maxrcheight) * 100.0;			
		if (regperc < 25.00) continue; // 65
		
		// rectangle(img_region, textrect, CV_RGB(0, 255, 0));
		// Rect* rect = new Rect(textrect.x, textrect.y, textrect.width, textrect.height);			
		// vaatame, et tekst ka liiga pisike pole, mida sellega teha ?
		if (textrect.height <= 6) continue;
		bgrects.push_back(textrect);					
	}

	// *******************************************************************
	// sorteerime Y telje järgi, jätame alles vaid suurima Yi
	// sort(bgrects.begin(), bgrects.end(), __compare);	
	// *******************************************************************

	std::sort(bgrects.begin(), bgrects.end(), sortbyY());
	bool processed = false;
	bool keepcleaning = false;
	//   for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
	while (!processed) {
		keepcleaning = false;
		for (unsigned int i = 0; i < bgrects.size(); i++) {
			cmp1 = bgrects[i];		
			// laius / kõrgus natuke liiga imelikud
			bool oddratio = (((float)cmp1.width > (float)cmp1.height * 2.2) || ((float)cmp1.height > (float)cmp1.width * 2.9));
			// liiga väike kontuur			
			if ((cmp1.width <= 4) || (cmp1.height <= 5) || oddratio) {			
				bgrects.erase(bgrects.begin() + i);
				keepcleaning = true;
				break;
			}

			// std::cout << "Char stat X: " << cmp1.x << " Y "<< cmp1.y << " width " << cmp1.width << " height " << cmp1.height  << endl;
			if (!keepcleaning)
			for (unsigned int j = 0; j < bgrects.size(); j++) {
				cmp2 = bgrects[j];
				//if  (((i != j) && (abs(cmp1.x - cmp2.x) <= 1) && (abs(cmp1.y - cmp2.y) < 3) && (cmp1.width <= cmp2.width)) || (cmp1.width <= 2) || (cmp1.height <= 2)) {
				if ((i != j) && (abs(cmp1.y - cmp2.y) <= 3) && (abs(cmp1.x - cmp2.x) <= 3)  && (cmp1.width <= cmp2.width)) {
					// bgrects.erase(cmp2);
					bgrects.erase(bgrects.begin() + j);
					keepcleaning = true;
					break;
				}
			}

			if (keepcleaning) break;
		}
		processed = !keepcleaning;
	}
	
	std::sort(bgrects.begin(), bgrects.end(), sortbyX());

	// *******************************************************************
	// Nüüd peame sealt veel eraldama kõige välimise ristküliku; ntx O ja D tähtedel on välimine ja sisemine ristkülik
	// *******************************************************************

	processed = false;
	keepcleaning = false;	
	while (!processed) {
		keepcleaning = false;
		for (unsigned int i = 0; i < bgrects.size(); i++) {
			cmp1 = bgrects[i];
			for (unsigned int j = 0; j < bgrects.size(); j++) {
				cmp2 = bgrects[j];				
				if ((i != j) && (cmp2.x >= cmp1.x && cmp2.x + cmp2.width <= cmp1.x + cmp1.width)) {
					keepcleaning = true;
					bgrects.erase(bgrects.begin() + j);
					break;
				}
			}
			if (keepcleaning) break;
		}
		processed = !keepcleaning;
	} 

	// Vähemalt kolm elementi peab olema samal tasemel +-4 (Y)	
	if (bgrects.size() > 0) {
		float avg = 0.00;
		for (unsigned int i = 0; i < bgrects.size(); i++) {
			cmp1 = bgrects[i];
			avg += cmp1.y;
		}

		avg = (float)avg / (float)bgrects.size();
		int samelevelcnt = 0;
		for (unsigned int i = 0; i < bgrects.size(); i++) {
			cmp1 = bgrects[i];
			// +-2 ei ole piisav, ntx kui numbrit pildistatud liiga nurga alt 
			if ((cmp1.y >= cvCeil(avg) - 4) && (cmp1.y <= cvCeil(avg) + 4)) {
				samelevelcnt++;
			}
		}
		if (samelevelcnt < 3) return false;
		// std::cout << avg << "\n";
	}

	// *******************************************************************
	// Vaatame, milline keskmine vahe kõrvutiseisvatel tähtedel, kui mingi täht järgmisest liiga kaugel, siis mingi jama	
	// TODO kas poleks parem deviationiga seda teha ?
	// *******************************************************************

	if (bgrects.size() > 0) {
		int avgx = 0.00, avgx1 = 0.00, avgx2 = 0.00;
		int i = 0, realcnt = 0;		

		// Proovime kaks keskmist läbi ! Esmalt keskelt paremale, siis vasakule
		// while (i < bgrects.size() - 1) {
		while (i < cvRound((float)bgrects.size() / 2) - 1) {
			cmp1 = bgrects[i];
			cmp2 = bgrects[i + 1];
			// std::cout << "--" << cmp1.x << " " << cmp2.x << "\n";
			int diff = (cmp2.x - cmp1.x);
			// Kas ei peaks liiga suured diffid ntx üle 35 välja viskama ?!?!
			if (diff > 35) {
				i += 2;
				continue;
			}		
			avgx += diff;
			i += 2;
			realcnt += 2;
		}

		// Akumulaator 1
		if (realcnt > 0.00) {
			avgx1 = (float)avgx / realcnt;
		}
		else {
			avgx1 = 0.00;
		}

		
		// Nüüd vasakult
		i = cvRound((float)bgrects.size() / 2);
		realcnt = 0;
		while (i < bgrects.size() - 1) {
			cmp1 = bgrects[i];
			cmp2 = bgrects[i + 1];
			// std::cout << "--" << cmp1.x << " " << cmp2.x << "\n";
			int diff = (cmp2.x - cmp1.x);
			// Kas ei peaks liiga suured diffid ntx üle 40 välja viskama ?!?!
			if (diff > 40) {
				i += 2;
				continue;
			}
			avgx += diff;
			i += 2;
			realcnt += 2;
		}

		// Akumulaator 2
		if (realcnt > 0.00) {
			avgx2 = (float)avgx / realcnt;
		}
		else {
			avgx2 = 0.00;
		}

		// Vaatame kumb akumulaator parem
		avgx = std::min(avgx1, avgx2);		
		// cout << "avgx "<< avgx << endl;		
		if ((avgx > 0.00) && (bgrects.size() > 1)) {			
			// std::cout << avgxf << "\n";
			int diffx = 0.00;
			// Enamasti esimene element imelik, ntx euro märk
			bool keepscanning = true;
			while (keepscanning) {
				if (bgrects.size() <= 1) break;
				cmp1 = bgrects[0];
				cmp2 = bgrects[1];
				diffx = cmp2.x - cmp1.x;
				// Esimene märk liiga kaugel
				if (diffx > (float)avgx * 2.2) {
					bgrects.erase(bgrects.begin());
					continue;
				}

				// Viimane märk liiga kaugel		
				/*
				cmp1 = bgrects[bgrects.size() - 2];
				cmp2 = bgrects[bgrects.size() - 1];
				if ((float)(cmp2.x - cmp1.x) > (float)avgx * 2.2) {
					bgrects.erase(bgrects.begin());
					continue;
				} */
				keepscanning = false;
			}			
		}				
	}

	// *******************************************************************
	// Teeme viimase analüüsi tuvastamaks, kas jäänud ristkülikud ei ole mingite anomaaliatega; ntx ühtivad piirkonnad
	// *******************************************************************	

	Mat img_prepared;
	img_prepared = useclahe ? image_clahe : img_region;
	
	// cout << " 0::: " << bgrects.size() << endl;
	removeElementsWithSamePos(bgrects);
	#ifdef COUTENABLED
	cout << " 1::: " << bgrects.size() << endl;
	#endif

	// *******************************************************************
	// Kõrguste "statistika"
	double avg_height = avgHeight2(bgrects);
	double avg_height_dev = heightDeviation(bgrects);
	
	double avg_width = avgWidth2(bgrects);
	double avg_width_dev = widthDeviation(bgrects);
	int char_diff_y = 0; // kas tähed on kuhugi suunas nihkes

	// *******************************************************************
	// Näiteks fail, elemendid mis järgi jäid on teineteisest liiga kaugel !!!
	// *******************************************************************
	// string filename = "C:/autonr/A130.jpg";
	// Natuke liiga hüperaktiivne funktsioon
	
	std::sort(bgrects.begin(), bgrects.end(), sortbyX());
	removeElementsWithOddDistance(img_prepared, bgrects, avg_height, avg_height_dev, avg_width, avg_width_dev);
	#ifdef COUTENABLED
	cout << " 2::: " << bgrects.size() << endl;
	#endif


	// *******************************************************************
	// Eemaldame kõik väikesed ja ülisuured ristkülikud
	// *******************************************************************
	
	removeElementsWithOddHW(bgrects, avg_height, avg_height_dev, avg_width, avg_width_dev);
	#ifdef COUTENABLED
	cout << " 3::: " << bgrects.size() << endl;	
	std::cout << " -- Avg height " << avg_height << " deviation " << avg_height_dev  << endl;
	#endif

	// *******************************************************************
	// Ntx 0 vaid keskmine osa, suurendame ala
	// *******************************************************************

	smartCharResize(img_prepared, bgrects, avg_height, avg_height_dev);
	#ifdef COUTENABLED
	cout << " 4::: " << bgrects.size() << endl;
	#endif

	// *******************************************************************
	// Vaatame, kui märke "puudu" proovime tuletada mingid positsioonid
	// *******************************************************************
	// std::sort(bgrects.begin(), bgrects.end(), sortbyX());
	char_diff_y = addPredictedRects(img_prepared, bgrects, avg_height, avg_height_dev, avg_width);
	#ifdef COUTENABLED
	cout << " 5::: " << bgrects.size() << endl;
	#endif

	// *******************************************************************
	// Viskame välja kõik märgid, kus mingi sürr värvus;
	// *******************************************************************
	int validrecs = validateImagePatterns(img_prepared, bgrects, useclahe) + 1;
	#ifdef COUTENABLED
	cout << " 6::: " << bgrects.size() << endl;
	#endif
	// @@		
	// teeme pildi 2x suuremaks !!
	/*
	Mat biggernr;
	cv::resize(img_region, biggernr, cv::Size(img_region.cols * 2, img_region.rows * 2), cv::INTER_NEAREST);
	imshow("_big_nr", biggernr);
	waitKey(0);
	return true; */

	// https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
	// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html	
	// cout << validrecs << endl;
	couldbenr = (bool)((validrecs >= 3) && (validrecs < 10)); //  (bgrects.size() >= 3);	
	if (couldbenr) {	
	// if (1==1) {
		std::sort(bgrects.begin(), bgrects.end(), sortbyX());

		// Nopime nüüd tähed välja ja saadame OCR kiu kallale		
		sendToOcrProcessing(img_region, img_negative, bgrects, char_diff_y, avg_height, avg_height_dev, avg_width);

		// cout << " char_diff_y " << char_diff_y << endl;
		#ifdef  TESTING		
		imwrite("c://temp//number_orig.jpg", img_region);

		stringstream wndname;
		for (unsigned int i = 0; i < bgrects.size(); i++) {
			textrect = bgrects[i];
			// rectangle(img_region, textrect, CV_RGB(0, 255, 0));
			rectangle(img_region, textrect, CV_RGB(0, 255, 0));
		}

		wndname << "MSER [area] " << _runindex << "_" << validrecs;
		imshow(wndname.str(), img_region);
		// imshow(wndname.str(), textImg);
		waitKey(0);
		#endif //  TESTING
	}

	mser.release();	

	// bool b = bboxes.size() > 0;
	return (bool)couldbenr;
}

/* ************************************************************************** */
// Eesti nr standardsuurus
// 520 x 113

bool processRegion(Mat sourceimg, RotatedRect mr, Rect rt, float aspect = DEFAULT_RATIOFNR, float deltacorr = DEFAULT_DELTACORR) { //  deltacorr = 1.2)
	int area = mr.size.height * mr.size.width;
	// jaburalt väikest suurust ei luba
	// if ((mr.size.width < 15.00) || (mr.size.height < 15.00) || (mr.size.width * 4 < mr.size.height)) return false;
	if ((mr.size.width < 15.00) || (mr.size.height < 15.00)) return false;	
	// vaatame kontuuri suhet
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
	r = (float)mr.size.height / (float)mr.size.width;
	// vaatame, et kontuur ei oleks liiga suure osakaaluga pildist 
	int sourceimgwidth = sourceimg.size().width;
	int regionwidth = rt.width;
	float asptoimg = ((float)regionwidth / (float)sourceimgwidth) * 100;
	// ***
	_runindex++; // ennekõike kasutame seda debugimisel, et mitmendat sektsiooni töötleme	
	
	if (asptoimg > 35) { 
		#ifdef COUTENABLED		
		std::cout << " Aspect to img > 35 FALSE: " << mr.angle << " " << rt.height << ":" << rt.width << " ratio: " << r << ":: " << asptoimg << " --- " << _runindex << "\n";		
		#endif
		//return false; 
	};
	
	// ca 29.2 x 89.38	
	// Teoreetiline pildi suhe			
	// if (_runindex == 2)  punane audi	
	if (1==1) {

		// std::cout << " --- " << _runindex << "\n";
		// rectangle(sourceimg, Point(rt.x, rt.y), Point(rt.x + rt.width, rt.y + rt.height), Scalar(0, 0, 255), 3, 8, 0);
	

		// if ((r >= aspect - deltacorr) && (r <= aspect + deltacorr)) {
		// Proovime nüüd leida, mustade ja valgete pixlite seose; valgeid peaks rohkem alati olema		
		// Mat current_roi = img_resized.rowRange(mr.y, mr.y + mr.height).colRange(mr.x, mr.x + mr.width);
		// teeme piirkonna suuremaks, et leida siiski nr. Tihti kolmnurka ei leita õigesti
		// Mat(contours[i]).convertTo(pointsf, CV_32F);
        // RotatedRect box = fitEllipse(pointsf);

		#ifdef COUTENABLED		
		std::cout << "Channels " << sourceimg.channels() << " ratio " << r << " aspect range " << aspect - deltacorr << " - " << aspect + deltacorr / 2 << endl;
		#endif
		
		int incareaby = 3; // pixels
		int rty = (int)rt.y - incareaby;
		int rtx = (int)rt.x - incareaby;
		float angle = (float)mr.angle;
		bool donothing = false;

		#ifdef COUTENABLED
		std::cout << " Angle: " << angle << " " << rt.height << ":" << rt.width <<  " ratio: "<< r  << ":: " << asptoimg << " --- " << _runindex <<"\n";
		#endif
		int cutimgheight = rt.y + rt.height + incareaby;
		int cutimgwidth = rt.x + rt.width + incareaby;
		if (cutimgwidth > sourceimg.size().width) {
			cutimgwidth = sourceimg.size().width;
		}
		if (cutimgheight > sourceimg.size().height) {
			cutimgheight = sourceimg.size().height;
		}

		// lõikame pildist ala meile parajaks
		Mat imgpart = sourceimg.rowRange((rty > 0 ? rty : 1), cutimgheight).colRange((rtx > 0 ? rtx : 1), cutimgwidth);		
		//if (_runindex == 7) // auto hoone ees
		if (1 == 1) {		
			float r = (float)mr.size.width / (float)mr.size.height;		
			if (r < 1) {			
				angle = (float)90.00 + angle;														
				#ifdef COUTENABLED
				std::cout << " New angle: " << angle << "\n";
				#endif
			} else {
				// mida me siin peaks tegema, sest ristküliku nurgad tihti arvutatakse siin väga valesti ?!?!				
				angle = 0.00;
				donothing = true;
			}

			// angle = 0.00;
			/*
			if (angle < 10) { // pole nii kriitiline nurk, et peaks reaalselt pöörama
				angle = 0.00;
			}*/

			int heightdv2 = (int)round(rt.height + incareaby / 2.00);
			int widthdv2 = (int)round(rt.width + incareaby / 2.00);
			cv::Point2f center = (widthdv2, heightdv2);


			Mat img_rotated;		
			Mat rotmat = getRotationMatrix2D(center, angle, 1);			
			// Korrigeerime numbrit, vahel sõltub pildistamisnurgast, siis ta nö kõver			
			warpAffine(imgpart, img_rotated, rotmat, imgpart.size(), CV_INTER_CUBIC);
			
			
			// warpAffine(imgpart, img_rotated, rotmat, Size(185,185), CV_INTER_CUBIC);
			/* 
			Mat img_crop;
			getRectSubPix(img_rotated, imgpart.size(), center1, img_crop);
			*/
			// Proovime sellest segadusest midagi asjalikku leida
			vector<Vec4i> nrlines;			
			Mat edges;
			Mat img_final_gray;
			Mat img_rotated_final;
			Mat img_rotated_final_hc;

			// imwrite("crop.jpg", image_roi);
			// imshow("Selected area", image_roi);
			cv::cvtColor(img_rotated, img_final_gray, CV_BGR2GRAY);
			cv::blur(img_final_gray, img_final_gray, Size(3, 3));
			
			// cvNot(img_final_gray, img_negative);
			// http://www.programming-techniques.com/2013/01/producing-nagative-of-grayscale-image-c.html
			// initialize the output matrix with zeros

			Mat img_negative = Mat::zeros(img_final_gray.size(), img_final_gray.type());
			// create a matrix with all elements equal to 255 for subtraction
			Mat sub_mat = Mat::ones(img_final_gray.size(), img_final_gray.type()) * 255;
			//subtract the original matrix by sub_mat to give the negative output new_image
			subtract(sub_mat, img_final_gray, img_negative);
			cv::Canny(img_negative, edges, 95, 100);
			
			/*
			stringstream wndname;
			wndname << " [__edges] " << _runindex << "____";
			imshow(wndname.str(), edges );
			waitKey(0); */

			// Canny(img_negative, edges, 35, 90, 3);
			// widthdv2
			// HoughLinesP(edges, nrlines, 1, CV_PI / 180, 15, 25, 5); // stable			
			// HoughLinesP(edges, nrlines, 1, CV_PI / 180, 15, 15, 5);

			int toppos = 0;
			int toplinelen = 0;
			int bottompos = 999999;
			int bottomlinelen = 0;
			int linelen = 0;
			int lineangle = 0.00;
			Point p1, p2;
			

			if (!donothing) {
				HoughLinesP(edges, nrlines, 1, CV_PI / 180, 15, 25, 5); // OK 25 px joone min pikkus !

				int heightbase = img_negative.size().height;
				heightdv2 = (int)round(heightbase / 4.00);

				// widthdv2 = (int)round(img_rotated.size().width / 2.00);
				// Mat Blank(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));	

				for (size_t i = 0; i < nrlines.size(); i++) {
					Vec4i l = nrlines[i];
					linelen = abs(l[0] - l[2]);
					p1 = Point(l[0], l[1]);
					p2 = Point(l[2], l[3]);
					//  atan2(y2 - y1, x2 - x1) * 180.0 / CV_P
					// Radiaan
					// https://stackoverflow.com/questions/24031701/how-can-i-determine-the-angle-a-line-found-by-houghlines-function-using-opencv
					// angle = atan2(p1.y - p2.y, p1.x - p2.x); // kas joon vasakul või paremale kaldu									
					// numbri ülemine osa, üritame leida piiri, mida saame kasutada
					if ((l[3] <= heightdv2) /*&& (abs(l[1] - l[3]) < 3)*/) {
						// line(img_rotated, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, CV_AA);
						if ((l[3] > toppos) && (linelen > toplinelen)) {
							//if ((lineangle == 0.00)) // || (lineangle > 179)) {
								lineangle = (float)(atan2(p1.y - p2.y, p1.x - p2.x) * 180.00) / CV_PI;
							//}							
							toppos = l[3];
							toplinelen = linelen;
							#ifdef DEBUG	
							line(img_negative, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
							#endif
						}
					}
					else if ((l[3] >= heightbase - heightdv2)/* && (abs(l[1] - l[3]) < 3)*/) {
						// line(img_rotated, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, CV_AA);
						if ((l[3] < bottompos) && (linelen > bottomlinelen)) {
							if ((lineangle == 0.00)) { //|| (lineangle > 179)) {
								lineangle = (float)(atan2(p1.y - p2.y, p1.x - p2.x) * 180.00) / CV_PI;
							}

							bottompos = l[3];
							bottomlinelen = linelen;
							#ifdef DEBUG
							line(img_negative, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
							#endif
						}
					}
					// line(imgpart, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, CV_AA);
				}
			}
			

			// Ei leidnud alt piirjooni
			if ((bottompos < 1) || (bottompos == 999999)) {
				//toppos = 0;
				bottompos = img_negative.size().height - 2;
			}


			Mat img_final_roi;			
			Mat image_roi;
			Mat img_rotated_step2;
			cv::Rect roi;

			bool correctedbylineangle = (bool)((lineangle != 0.00) && (abs(lineangle) != 180));			
			// Vaatame täiendavalt üle, kas korrigeerida. Sest vahel ristküliku järgi korrigeeritakse valesti
			if (correctedbylineangle) {
				// v1
				heightdv2 = (int)round(img_negative.size().height / 2.00);
				widthdv2 = (int)round(img_negative.size().width / 2.00);
				cv::Point2f center2 = (widthdv2, heightdv2);
				float newangle = (float)180 - lineangle;
				newangle = newangle - angle;
				#ifdef COUTENABLED
				std::cout << " Found line angle: " << lineangle << " 180 - lineangle " << 180 - lineangle << " rectangle angle " << angle << " new angle " << newangle << " \n";
				#endif
				if (abs(newangle) <= 25) {								
					Mat rotmat = getRotationMatrix2D(center2, 0.215 , 1);
					Mat img_corrected;

					warpAffine(img_negative, img_corrected, rotmat, img_negative.size(), CV_INTER_CUBIC);
					img_negative = img_corrected;
					// v2
					heightdv2 = (int)round(img_rotated.size().height / 2.00);
					widthdv2 = (int)round(img_rotated.size().width / 2.00);
					cv::Point2f center3 = (widthdv2, heightdv2);
					rotmat = getRotationMatrix2D(center3, newangle, 1);
					warpAffine(img_rotated, img_corrected, rotmat, img_rotated.size(), CV_INTER_CUBIC);
					img_rotated = img_corrected;
				}
				else {
					correctedbylineangle = false;
				}
			}

			// double Angle = atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
			// Just ignore the line which has angle 90 degree(vertical line).
			if (!correctedbylineangle) {
				// Rect roi(toppos, toppos, img_rotated.size().width, (bottompos - toppos) + 1);				
				roi.x = 0;
				roi.y = (toppos > 0 ? toppos : 0);
				roi.width = img_negative.size().width - 1;
				roi.height = (bottompos - toppos) + 1;
				if (roi.height < 1) {
					roi.height = img_negative.size().height - 1;
				}

				image_roi = img_negative(roi);
				img_rotated_step2 = img_rotated(roi);


				img_rotated_final = img_rotated_step2;
				img_final_roi = image_roi;
			}
			else {
				img_rotated_final = img_rotated;
				img_rotated_step2 = img_rotated;
				image_roi = img_negative;
				img_final_roi = image_roi;
			}

			/* 
			Üldiselt kõik tore, aga vasakule jääb euromärk, see hele ala, teeme dummy lahenduse, 
			proovime leida 3 tumedat pixlit samal distansil, väldime keerukaid algoritme			
			*/
			int x = 0, y = 0;
			try
			{
				int camiddle = (int)cvRound(image_roi.rows / 2); // (int)cvRound(image_roi.size().height / 2);
				int startpos = (camiddle - 3 > 0 ? camiddle - 3: 1);
				//int endpos = (image_roi.rows + 4 < image_roi.rows ? camiddle + 4 : image_roi.rows);				
				int endpos = (image_roi.rows + 4 > image_roi.rows ? camiddle + 4 : image_roi.rows);				
				int vcolorcnt = 0;
				bool reroi = false;				
				for (x = 0; x < image_roi.cols; x++) {
					vcolorcnt = 0;
					for (y = startpos; y < endpos; y++) {					
						Scalar colour = image_roi.at<uchar>(Point(x, y));
						// Vec3b,  Vec2b, Vec						
						//color = (uchar)(130); // 85 tumehall / 130 ka veel suht tume / 140 helehall
						// cout << "--" << colour.val[0]  << "\n";						
						if (colour.val[0] < 130) {							
							// muudame pixlite värvi
							// uchar & color = image_roi.at<uchar>(y, x);
							// color = (uchar)(10);						
							vcolorcnt++;							
						}
						// image_roi.at<Vec3b>(Point(x, y)) = 168;
						// Vec3b & color = image_roi.at<Vec3b>(y, x);
						// color[0] = 158;
						// image_lt.at<cv::Vec3b>(x, y)[0] = 178;  //turn the pixel value @ (k,i) to yellow (0,255,255)
						// image_lt.at<cv::Vec3b>(x, y)[1] = 255;
						// image_lt.at<cv::Vec3b>(x, y)[2] = 255;
						// image_roi.at< unsigned int >(x, y) = 0x41CA0000;
						// cout << "x " << x << "y" << y << " color " << colour.val[0] << "\n";						
					}

					if (vcolorcnt > 6) {
						reroi = true;
						#ifdef COUTENABLED
						cout << "Found dark color area at x " << x << " y " << y << "\n";
						#endif // DEBUG											
						break;
					}
				}
				
				// Lõikame vasakult ääre ka ära, seal ntx EURO märk sinisel taustal, kas peaks seda tuvastama ?!?!				
				if ((reroi) && (x <= cvRound(image_roi.size().width / 2.00)) && !donothing) {
					roi.x = x;
					roi.y = 0;
					roi.width = image_roi.size().width - x;
					roi.height = image_roi.size().height;			
					img_final_roi = image_roi(roi);
					img_rotated_final = img_rotated_final(roi);										
				}
			}
			catch (Exception *e)
			{
				#ifdef COUTENABLED
				cout << "Find color exception x " << x << " y " << y << e->msg << endl;
				#endif
			} 
					
			#ifdef DEBUG
			stringstream ss(stringstream::in | stringstream::out);				
			// img_rotated_final.convertTo(img_rotated_final_hc, -1, 2, 0); //increase the contrast (double)
			// ss << "c://temp//licplate_v2_" << i << ".jpg";
			ss << "c://temp//licplate_v2" << ".jpg";
			// imwrite(ss.str(), img_final_gray);
			imwrite(ss.str(), img_rotated_final);			
			imshow("Rectangle block", imgpart);
			imshow("Car nr rotated", img_rotated_final);
			imshow("Inverted", img_negative);		
			waitKey(0);
			// imshow("Negative", img_negative);			
			#endif
			
			#ifdef COUTENABLED
			std::cout << " FOUND: " << _runindex << "\n";
			//std::cout << " FOUND: " << r << " darkperc " << darkperc << " brightperc " << brightperc << " loop index " << _runindex << "\n";
			#endif
			
			#ifdef DEBUG
			stringstream wndname;
			wndname << "Found nr [area] " << _runindex;
			imshow(wndname.str(), img_rotated_final);
			#endif // DEBUG

						
			bool b = regionhasText2(img_rotated_final, img_final_roi);
			if (b) {
				// imshow("TEST", img_rotated_final);
				// waitKey(0);
			}
			
			#ifdef SHOWFOUNDNUMBER
			/*
			if (b) {
				stringstream wndname;
				wndname << "Found nr [area] " << _runindex;
				imshow(wndname.str(), img_rotated_final);
			} */			
			#endif
			return b;
		}
		
	}
	return false;
}

/* ************************************************************************** */

int possibleMergeItem(vector<vector<Point> > sccontours, int currindex) {
	int diffx, diffy;
	Rect neighbor, currrent = boundingRect(sccontours[currindex]);
	/*
	if ((currrent.width < 45) || (currrent.height > 120)) 
		return -1;	
	*/
	// int i = 252;
	for (int i = 0; i < sccontours.size(); i++)
		if (i != currindex) {
			// RotatedRect minRect = minAreaRect(pointsInterest);		
			neighbor = boundingRect(sccontours[i]);
			diffx = (int)currrent.x + (int)currrent.width;
			diffx = abs((int)neighbor.x - diffx);
			//  cout << " DIFFX " << diffx << endl;
			if (diffx > 45)
				continue;
				
			diffy = abs((int)neighbor.y - (int)currrent.y);
			// cout << " DIFFY " << diffy << endl;
			//if (diffy > 21) 
			if (diffy > 32)
				continue;

			// cout << " MERGE: " << diffx << " " << diffy << endl;
			if ((neighbor.height >= 36) && (neighbor.width > 55) && ((int)neighbor.width < (int)(currrent.width * 2))) {
				// cout << " OUT " << i << endl;
				return i;
			}
		}
	return -1;
}

/* ************************************************************************** */

Rect mergeRegion(Rect area1, Rect area2) {
	Rect am;
	int x, y, height, width;
	x = area1.x;
	y = area1.y;
	if (area2.x < x) {
		x = area2.x;
	}
	if (area2.y < y) {
		y = area2.y;
	}
	am.x = x;
	am.y = y;
	height = (int)abs(area2.y - area1.y) + area1.height + 1;
	width = (int)(area2.x - area1.x) + area1.width + 1;
	am.height = height;
	am.width = width;
	return am;	
}

/* ************************************************************************** */
// Lõikame regiooni väiksemaks, et töötlemine oleks kiirem.
/* ************************************************************************** */
// Peaks vist laiuse trucate ka tegema ?
Mat truncateImage(Mat image, int perctop, int percbottom) {
	if ((perctop < 0) || (percbottom < 0) || ((perctop + percbottom) > 85)) {
		return image;
	}

	double newy = 0.00;
	int diff=0, x=0, newheight=0, currheight = image.size().height, currwidth = image.size().width;
	newy = perctop * currheight / 100.0;	
	newheight = (int)currheight - cvRound(newy);	
	if (percbottom > 0) {		
		newheight = newheight - cvRound(percbottom * newheight / 100);				
	}

	// cout << diff << x << " " << newy << " "<< cvRound(newy) << " " << " " << currheight <<  " " << newheight << " " << currwidth << endl;
	Mat rez = image(Rect(x, newy, currwidth, newheight));	
	return rez;
}

/* ************************************************************************** */
// TODO refact !!!
/* ************************************************************************** */

bool processImage(Mat &image, bool usesobel = true, float ratio = DEFAULT_RATIOFNR, float ratio_delta_corr = DEFAULT_DELTACORR) {
	Mat grey, temp;
	int foundnrs;
	Rect temprect;
	vector<vector<Point> > contours;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy;
	//vector<RotatedRect> regions_of_interest;
	vector<Rect> regions_of_interest;
	//vector<Vec4i> hierarchy;
	vector<vector<Point> > contours_poly;
	vector<vector<Point> > approx;
	Mat img_resized, img_sobel;		
	auto fnc_start_time = std::chrono::high_resolution_clock::now();
	
	// resize(image, img_resized, Size(520, 410));
	// img_resized = image.clone();
	// float x_scale = static_cast<float>(image.cols) / static_cast<float>(img_resized.cols);
	// float y_scale = static_cast<float>(image.rows) / static_cast<float>(img_resized.rows);
	foundnrs = 0;
	// cvtColor(img_resized, grey, CV_BGR2GRAY);
	// Sobel(grey, img_resized, CV_16S, 1, 0, 3);


	// img_resized.convertTo(img_higher_contrast, -1, 2, 0); //increase the contrast (double)
	// img_resized.convertTo(img_higher_contrast, -1, 1, 20); //increase the brightness by 20 for each pixel 
	// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html	
	// img_higher_contrast = image.clone();
	// mis oleks optimaalne suurus ? Reserveerime selle koha
	img_resized = image;

	// ----------------------------------------------------------------------------

	//img_resized = eqHistogram(img_resized);
	cvtColor(img_resized, grey, CV_BGR2GRAY);
	grey = eqHistogram(grey);

#ifdef DEBUG
	imshow("Balanced gray", grey);
#endif

	// cvtColor(img_resized, grey, CV_BGR2GRAY);
	// blur(grey, grey, Size(5, 5));
	blur(grey, grey, Size(5, 5)); // parim väärtus 5	
	if (usesobel) {
		// kuidas valida, mis hetkel kumba kasutada...
		Sobel(grey, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		// https://stackoverflow.com/questions/37372410/opencv-convertscaleabs-in-sobel
		convertScaleAbs(img_sobel, img_sobel, 1);
	}
	else {
		Canny(grey, img_sobel, 50, 200, 3);
	}

	auto start_time = std::chrono::high_resolution_clock::now();
	//threshold image
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
#ifdef DEBUG
	imshow("Threshold", img_threshold);
#endif
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	std::chrono::duration<double> elapsed = end_time - start_time;
#ifdef COUTENABLED
	std::cout << "Threshold elapsed time: " << elapsed.count() << " s\n";
#endif


	start_time = std::chrono::high_resolution_clock::now();
	// std::cout << "Threshold took " << (std::chrono::duration_cast<std::chrono::microseconds>(time).count()) / 1000 << " to run.\n";
	// Morphplogic operation close
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);

#ifdef DEBUG
	imshow("MorphologyEx", img_threshold);
#endif // DEBUG

	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	elapsed = end_time - start_time;

#ifdef COUTENABLED
	std::cout << "MorphologyEx elapsed time: " << elapsed.count() << " s\n";
#endif


	start_time = std::chrono::high_resolution_clock::now();

	vector< Vec4i > nrhierarchy;
	vector< vector <Point> > nrcontours;

	start_time = std::chrono::high_resolution_clock::now();
	findContours(img_threshold,
		nrcontours, // a vector of contours
		nrhierarchy,
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // all pixels of each contours

	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	elapsed = end_time - start_time;

	#ifdef COUTENABLED
	std::cout << "findContours elapsed time: " << elapsed.count() << " s\n";
	#endif	
	RotatedRect asrotrect;
	Rect asrect;
	vector<vector<Point> > allcontours;
	int mergeindex = 0;
	// Normaliseerime: kõik kontuurid ühte nimistusse, et poleks mingit segadust ! TODO eelkoristus
	for (int i = 0; i < nrcontours.size(); i = nrhierarchy[i][0]) { // iterate through each contour.
		allcontours.push_back(nrcontours[i]);
	}

	// Käime läbi kontuurid ja võimaluse korral mergeme	
	for (int i = 0; i < allcontours.size(); i++) {			 
		// RotatedRect minRect = minAreaRect(pointsInterest);		
		asrotrect = minAreaRect(allcontours[i]);
		asrect = boundingRect(allcontours[i]);
		// if ((abs(asrotrect.angle) < 10) || (abs(asrotrect.angle) < 100 && abs(asrotrect.angle) > 85))		
		if (1 == 1) {
			bool b = false;
			mergeindex = possibleMergeItem(allcontours, i);
			// REFACT: defragmenteerunud koht, proovime kokku põimida
			if (mergeindex > 0) {
				Rect mergeasrect = mergeRegion(asrect, boundingRect(allcontours[mergeindex]));
				asrotrect.size.width = mergeasrect.width;
				asrotrect.size.height = mergeasrect.height;
				// rectangle(img_resized, Point(mergeasrect.x, mergeasrect.y), Point(mergeasrect.x + mergeasrect.width, mergeasrect.y + mergeasrect.height), Scalar(0, 0, 255), 3, 8, 0);
				b = (processRegion(img_resized, asrotrect, mergeasrect, ratio, ratio_delta_corr));			
				if (b) {
					rectangle(img_resized, Point(mergeasrect.x, mergeasrect.y), Point(mergeasrect.x + mergeasrect.width, mergeasrect.y + mergeasrect.height), Scalar(0, 0, 255), 3, 8, 0);
					foundnrs++;
				}
			}

			if ((!b) && (processRegion(img_resized, asrotrect, asrect, ratio, ratio_delta_corr))) {
				rectangle(img_resized, Point(asrect.x, asrect.y), Point(asrect.x + asrect.width, asrect.y + asrect.height), Scalar(0, 0, 255), 3, 8, 0);
				foundnrs++;
			}
		}
	}
	/*
	imshow("---- img_resized --- ", img_resized);
	waitKey(0);
	*/
	#ifdef DEBUG
	imshow("---- img_sobel --- ", img_sobel);
	imshow("---- img_resized --- ", img_resized);
	waitKey(0);
	#endif // DEBUG
	
	// Sleep(10000);
	auto fnc_end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_total = fnc_end_time - fnc_start_time;
	cout << ">> Elapsed time: " << elapsed_total.count() << " " << endl;
	cout << ">> Found: " << foundnrs << " possible candidates " << endl;

	// kuvatakse leitud regioonid
	#ifdef SHOW_FOUNDNUMBER
	if (foundnrs > 0) {
		Mat img_smaller;
		if ((img_resized.size().width > 1400) || (img_resized.size().height > 1200)) {
			cv::resize(img_resized, img_smaller, cv::Size(1200, 800), cv::INTER_NEAREST);
		}
		else {
			img_smaller = img_resized;
		}
		imshow("Scanned image", img_smaller);
		waitKey(0);
	}		
	#endif
	return (bool)foundnrs > 0;
}

/* ************************************************************************** */

bool processImages(string filename) {
	// Teksti eelfilter 
	// http://study.marearts.com/2015_06_07_archive.html
	// er_filter1 = text::createERFilterNM1(text::loadClassifierNM1("C:/Tarkvara/Opencv/opencv_contrib/modules/text/samples/trained_classifierNM1.xml"),
	//	16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
		
	#ifdef TESTING
	er_filter1 = text::createERFilterNM1(text::loadClassifierNM1("C:/Projektid/Research/x64/Debug/trained_classifierNM1.xml"), 4, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	er_filter2 = text::createERFilterNM2(text::loadClassifierNM2("C:/Projektid/Research/x64/Debug/trained_classifierNM2.xml"), 0.5);
	#else
	// NB me ei kasuta seda hetkel kusagil, liiga aeglane oli !!!
	// er_filter1 = text::createERFilterNM1(text::loadClassifierNM1("trained_classifierNM1.xml"), 4, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	// er_filter2 = text::createERFilterNM2(text::loadClassifierNM2("trained_classifierNM2.xml"), 0.5);
	#endif

	Mat image = imread(filename, 0);
	image = imread(filename, 1);
	if (image.empty()) {
		printf("Error loading image: %s\n", filename.c_str());
		return false;
	}


	// Alustame pildi töötlusega	
	image = truncateImage(image, _perctop, _percbottom);
	bool fndimg = processImage(image, false, DEFAULT_RATIOFNR_LOOSE, DEFAULT_DELTACORR_LOOSE);
	if (!fndimg) {
		cout << ">> Second try...." << endl;
		fndimg = processImage(image, true, DEFAULT_RATIOFNR_LOOSE, DEFAULT_DELTACORR_LOOSE);
	}

	// Hetkel me ei kasuta seda
	// memory clean-up
	// er_filter1.release();
	// er_filter2.release();
	return true;
}

/* ************************************************************************** */
// Kindla frame pealt lugemine
// cap.set(CV_CAP_PROP_POS_FRAMES, 15); //Set index to 0 (start frame)
bool processVideo(string filename) {
	cv::VideoCapture video(filename);	
	if (!video.isOpened())  // isOpened() returns true if capturing has been initialized.
	{
		cout << "Cannot open the video file. \n";
		return -1;
	}

	int W = video.get(CV_CAP_PROP_FRAME_WIDTH);
	int H = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	int count = video.get(CV_CAP_PROP_FRAME_COUNT);
	int currentframe = 0;
	double fps = video.get(CV_CAP_PROP_FPS);
	namedWindow("LVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	// first argument: name of the window.
	// second argument: flag- types: 
	// WINDOW_NORMAL : The user can resize the window.
	// WINDOW_AUTOSIZE : The window size is automatically adjusted to fitvthe displayed image() ), and you cannot change the window size manually.
	// WINDOW_OPENGL : The window will be created with OpenGL support.
	while (1) {
		// kas pilti juba vähendatud		
		Mat frame;
		// Mat object is a basic image container. frame is an object of Mat.
		if (!video.read(frame)) { // if not success, break loop read() decodes and captures the next frame.		
			cout << "\n Cannot read the video file. \n";
			break;
		}
		// capture.get(CV_CAP_PROP_FRAME_COUNT); // retuns the number of total frames 
		currentframe = video.get(CV_CAP_PROP_POS_FRAMES);
		frame = truncateImage(frame, _perctop, _percbottom);
		imshow("LVideo", frame);
		// first argument: name of the window.
		// second argument: image to be shown(Mat object).

		if (waitKey(30) == 27) { // Wait for 'esc' key press to exit		
			break;
		}
	}

	video.release();
	// Closes all the frames
	destroyAllWindows();
	return true;
}
/* ************************************************************************** */

int main( int argc, const char** argv )
{    	
	// https://stackoverflow.com/questions/13391252/how-to-print-latin-characters-to-the-c-console-properly-on-windows
	typedef std::codecvt_byname<wchar_t, char, std::mbstate_t> codecvt;
	// the following relies on non-standard behavior, codecvt destructors are supposed to be protected and unusable here, but VC++ doesn't complain.
	std::wstring_convert<codecvt> cp1252(new codecvt(".1252"));
	std::wstring_convert<codecvt> cp850(new codecvt(".850"));
	// https://www.riigiteataja.ee/akt/13040616
	std::cout << "***************************************************************************" << endl;
	std::cout << "*** Estonian license plate detector beta (supported types (A1, A2, A8, E2))" << endl;
	std::cout << cp850.to_bytes(cp1252.from_bytes("*** Copyright Ingmar Tammeväli 2019 www.stiigo.com")).c_str() << endl;
	std::cout << "***************************************************************************" << endl;
	std::cout << endl;
	
	#ifdef TESTING
	vector <string> testfiles;
	testfiles.push_back(TESTFILEV);
	for (unsigned i = 0; i < testfiles.size(); i++) {
		processImages(testfiles[i]);
	}

	cout << "DONE !";
	// string key;
	// cin >> key;
	std::cin.ignore();
	#else
	const cv::String keys = { 
		"{help h | | Help}"
		"{@filename | | Picture file }"
		"{od | outputdir | <none> | Output directory}"
		"{cf | | Number plate color check}"
		"{sf | | Show found areas separately}"
		"{wd | video | <none> | Video file to scan}"
		"{pt | ptop |15 | Percent from top}"
		"{pb | pbottom |5 | Percent from bottom}"
		// "{wt | | Write areas to file}"
	};
	// "{fps            | -1.0 | fps for output video }"
	// "{N count        |100   | count of objects     }"
	// "{ts timestamp   |      | use time stamp       }"
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		std::cout << "Filename <params> " << endl << endl;
		std::cout << "  --od	Output directory (example --od=c:\\temp\\ )" << endl;
		std::cout << "  --cf	Heuristic number plate color analyzer" << endl;
		std::cout << "  --sf	Show found areas separately" << endl;
		std::cout << "  --wd	Video file to scan (example --wd=c:\\temp\\myfile.mp4)" << endl;
		std::cout << "  --pt	Percent from top (example --pt=15)" << endl;
		std::cout << "  --pb	Percent from bottom (example --pb=5)" << endl;
		//std::cout << "  --wt	Write areas to file" << endl;
		std::cout << endl;
		return 0;
	}

	if (argc < 2) {
		cout << "Filename is missing !" << endl;
		return 0;
	}
	

	string filename = parser.get<string>(0);
	_outputdir = parser.get<string>("od");
	_videofile = parser.get<string>("wd");
	//_outputdir = parser.get<String>(1);	
	_allowcolorfilter = parser.has("cf");
	_showfoundareas = parser.has("sf");
	if ((parser.get<string>("pt") != "pt") && (parser.get<string>("pt") != "ptop")) {
		_perctop = parser.get<int>("pt");
	}
	
	if ((parser.get<string>("pb") != "pb") && (parser.get<string>("pb") != "pbottom"))  {
		_percbottom = parser.get<int>("pb");
	}
		
	/*
	if (parser.has("checkcolor")) {
		_allowcolorfilter = parser.get<String>("checkcolor") == "true";
		cout << "varvi filter olemas " << " " << parser.get<String>("checkcolor") << endl;
	}*/

	cout << ">> Detecting...." << endl;
	if ((_outputdir != "od") && (_outputdir != "outputdir")) {
		cout << ">> Output directory " << _outputdir << endl;
	}

	_videoprocessing = (bool)((_videofile != "wd") && (_videofile != "video"));
	if (_videoprocessing) {
		processVideo(_videofile);		
	}
	else {
		processImages(filename);
	}


	
	
	/*
	int N = parser.get<int>("N");
	double fps = parser.get<double>("fps");
	String path = parser.get<String>("path");
	use_time_stamp = parser.has("timestamp");
	String img1 = parser.get<String>(0);
	String img2 = parser.get<String>(1);
	*/
	#endif

    return 0;
}


